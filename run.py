#!/usr/bin/env python3
"""
Weather Forecaster System - Main Entry Point

This script orchestrates the Multi-Agent System (MAS) pipeline for weather prediction.
It can be run in different modes:
    - full: Complete pipeline (data retrieval, training, evaluation, prediction)
    - train: Train models only
    - predict: Make predictions with existing models
    - evaluate: Evaluate existing models
    - ui: Launch Streamlit UI

Usage:
    python run.py --mode full --location "Warsaw" --days 365
    python run.py --mode train --epochs 30
    python run.py --mode predict
    python run.py --mode ui
"""

import argparse
import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
import logging

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np

from src.utils.config import Config
from src.utils.helpers import set_seed, get_device

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Weather Forecaster - MAS Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--mode",
        type=str,
        default="full",
        choices=["full", "train", "predict", "evaluate", "ui"],
        help="Execution mode"
    )
    
    parser.add_argument(
        "--location",
        type=str,
        default="Warsaw",
        help="Location name for weather data"
    )
    
    parser.add_argument(
        "--latitude",
        type=float,
        default=52.2297,
        help="Latitude for weather data"
    )
    
    parser.add_argument(
        "--longitude",
        type=float,
        default=21.0122,
        help="Longitude for weather data"
    )
    
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="Days of historical data to retrieve"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Number of training epochs"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size"
    )
    
    parser.add_argument(
        "--seq-length",
        type=int,
        default=7,
        help="Sequence length for predictions"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    
    parser.add_argument(
        "--no-train-baseline",
        action="store_true",
        help="Skip baseline model training"
    )
    
    parser.add_argument(
        "--no-train-transformer",
        action="store_true",
        help="Skip transformer model training"
    )
    
    return parser.parse_args()


def run_data_retrieval(args, config: Config) -> Path:
    """
    Run data retrieval agent.
    
    Returns:
        Path to the saved CSV file
    """
    from src.agents.data_retriever import DataRetriever
    
    logger.info("=" * 60)
    logger.info("PHASE 1: DATA RETRIEVAL")
    logger.info("=" * 60)
    
    retriever = DataRetriever(
        latitude=args.latitude,
        longitude=args.longitude
    )
    
    # Calculate date range
    end_date = datetime.now() - timedelta(days=1)  # Yesterday
    start_date = end_date - timedelta(days=args.days)
    
    logger.info(f"Fetching weather data for {args.location}")
    logger.info(f"Date range: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
    
    # Fetch historical data
    df = retriever.fetch_historical_data(
        start_date=start_date.strftime("%Y-%m-%d"),
        end_date=end_date.strftime("%Y-%m-%d"),
        daily=True
    )
    
    if df is None or df.empty:
        raise RuntimeError("Failed to retrieve weather data")
    
    logger.info(f"Retrieved {len(df)} days of weather data")
    logger.info(f"Features: {list(df.columns)}")
    
    # Save data
    config.create_directories()
    data_path = config.raw_data_dir / f"weather_data_{args.location.lower()}.csv"
    df.to_csv(data_path, index=False)
    
    return data_path


def run_data_processing(data_path: Path, config: Config):
    """
    Run data processing agent.
    
    Returns:
        Tuple of (splits_dict, loaders_dict, data_agent)
    """
    from src.agents.data_agent import DataAgent
    
    logger.info("=" * 60)
    logger.info("PHASE 2: DATA PROCESSING")
    logger.info("=" * 60)
    
    data_agent = DataAgent(config=config)
    
    # Full data preparation pipeline
    splits, loaders = data_agent.prepare_data(
        filepath=data_path,
        batch_size=config.training.batch_size
    )
    
    logger.info(f"Train sequences: {len(splits['train'])}")
    logger.info(f"Val sequences: {len(splits['val'])}")
    logger.info(f"Test sequences: {len(splits['test'])}")
    
    # Save the scaler for inference
    scaler_path = config.models_dir / "scaler.joblib"
    data_agent.save_scaler(scaler_path)
    
    return splits, loaders, data_agent


def run_baseline_training(splits, config: Config):
    """
    Train baseline models.
    
    Returns:
        Trained BaselineAgent
    """
    from src.agents.baseline_agent import BaselineAgent
    
    logger.info("=" * 60)
    logger.info("PHASE 3: BASELINE MODEL TRAINING")
    logger.info("=" * 60)
    
    # Use RandomForest as it handles non-contiguous class labels better
    baseline_agent = BaselineAgent(use_xgboost=False, random_state=config.seed)
    
    train_split = splits['train']
    val_split = splits['val']
    
    # Prepare data for sklearn (flatten sequences) - convert tensors to numpy
    X_train = train_split.sequences.numpy().reshape(len(train_split), -1)
    y_temp_train = train_split.temp_targets.numpy().ravel()
    y_weather_train = train_split.class_targets.numpy()
    
    X_val = val_split.sequences.numpy().reshape(len(val_split), -1)
    y_temp_val = val_split.temp_targets.numpy().ravel()
    y_weather_val = val_split.class_targets.numpy()
    
    # Train
    results = baseline_agent.train(
        X_train, y_temp_train, y_weather_train,
        X_val, y_temp_val, y_weather_val
    )
    
    logger.info("Baseline Model Training Results:")
    if 'val' in results:
        for name, value in results['val'].items():
            if isinstance(value, float):
                logger.info(f"  {name}: {value:.4f}")
    
    # Save model (baseline saves to directory)
    model_path = config.models_dir / "baseline"
    baseline_agent.save_models(model_path)
    
    return baseline_agent


def run_transformer_training(loaders, config: Config, args):
    """
    Train transformer model.
    
    Returns:
        Trained TransformerAgent
    """
    from src.agents.transformer_agent import TransformerAgent
    
    logger.info("=" * 60)
    logger.info("PHASE 4: TRANSFORMER MODEL TRAINING")
    logger.info("=" * 60)
    
    # Create agent with transformer config
    transformer_agent = TransformerAgent(
        num_features=len(config.data.features),
        d_model=config.transformer.d_model,
        num_heads=config.transformer.num_heads,
        num_layers=config.transformer.num_layers,
        d_ff=config.transformer.d_ff,
        dropout=config.transformer.dropout,
        max_seq_len=config.transformer.max_seq_len,
        num_classes=len(config.data.weather_classes),
        seed=config.seed
    )
    
    # Train using DataLoaders
    history = transformer_agent.train(
        train_loader=loaders['train'],
        val_loader=loaders['val'],
        num_epochs=args.epochs,
        learning_rate=config.training.learning_rate,
        early_stopping_patience=config.training.early_stopping_patience
    )
    
    # Log training history
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {history.best_val_loss:.4f} at epoch {history.best_epoch}")
    
    # Save model
    model_path = config.models_dir / "transformer_model.pth"
    transformer_agent.save(model_path)
    
    return transformer_agent


def run_evaluation(baseline_agent, transformer_agent, splits, loaders, config: Config):
    """
    Run evaluation on test data.
    
    Returns:
        Evaluation results dictionary
    """
    logger.info("=" * 60)
    logger.info("PHASE 5: MODEL EVALUATION")
    logger.info("=" * 60)
    
    test_split = splits['test']
    
    # Prepare test data
    X_test = test_split.sequences.numpy()
    X_test_flat = X_test.reshape(len(X_test), -1)
    y_temp_test = test_split.temp_targets.numpy().ravel()
    y_weather_test = test_split.class_targets.numpy()
    
    # Evaluate baseline
    baseline_metrics = baseline_agent.evaluate(X_test_flat, y_temp_test, y_weather_test)
    
    logger.info("\nBaseline Model Test Metrics:")
    for name, value in baseline_metrics.items():
        if isinstance(value, float):
            logger.info(f"  {name}: {value:.4f}")
    
    # Evaluate transformer
    transformer_metrics = transformer_agent.evaluate(loaders['test'])
    
    logger.info("\nTransformer Model Test Metrics:")
    logger.info(f"  MAE: {transformer_metrics.metrics.get('temp_MAE', 0):.4f}")
    logger.info(f"  Accuracy: {transformer_metrics.metrics.get('weather_Accuracy', 0):.4f}")
    logger.info(f"  F1 Score: {transformer_metrics.metrics.get('weather_F1_macro', 0):.4f}")
    
    # Extract baseline metrics properly
    baseline_reg = baseline_metrics['regression'].metrics
    baseline_cls = baseline_metrics['classification'].metrics
    baseline_flat = {
        'temp_MAE': baseline_reg.get('MAE', 0),
        'temp_RMSE': baseline_reg.get('RMSE', 0),
        'temp_R2': baseline_reg.get('R2', 0),
        'weather_Accuracy': baseline_cls.get('Accuracy', 0),
        'weather_F1_macro': baseline_cls.get('F1_macro', 0),
        'weather_F1_weighted': baseline_cls.get('F1_weighted', 0)
    }
    
    # Save metrics to JSON file
    comparison_results = {
        'timestamp': datetime.now().isoformat(),
        'baseline': baseline_flat,
        'transformer': transformer_metrics.metrics,
        'comparison': {
            'temp_winner': 'Transformer' if transformer_metrics.metrics.get('temp_MAE', 999) < baseline_flat.get('temp_MAE', 999) else 'Baseline',
            'weather_winner': 'Transformer' if transformer_metrics.metrics.get('weather_Accuracy', 0) > baseline_flat.get('weather_Accuracy', 0) else 'Baseline'
        }
    }
    
    metrics_path = config.models_dir / "metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(comparison_results, f, indent=2, default=str)
    logger.info(f"\n📊 Saved metrics to {metrics_path}")
    
    return comparison_results


def run_prediction(transformer_agent, test_split, config: Config):
    """
    Make predictions using the model.
    
    Returns:
        Prediction results
    """
    from src.agents.narrator_agent import NarratorAgent
    
    logger.info("=" * 60)
    logger.info("PHASE 6: PREDICTION")
    logger.info("=" * 60)
    
    # Use last sequence from test data
    recent_sequence = test_split.sequences[-1:].clone()
    
    # Make prediction
    predictions = transformer_agent.predict(recent_sequence)
    
    temp_pred = predictions['temperature']
    weather_pred = predictions['weather_type']
    weather_probs = predictions['class_probabilities']
    
    logger.info(f"Predicted Temperature: {temp_pred:.1f}°C")
    logger.info(f"Predicted Weather: {weather_pred}")
    logger.info(f"Weather Probabilities: {weather_probs}")
    
    # Generate narrative
    narrator = NarratorAgent()
    # Get the confidence for the predicted weather type
    confidence = weather_probs.get(weather_pred, 0.8)
    is_cold = predictions.get('is_cold_day', temp_pred < 10)
    
    narrative = narrator.generate_forecast(
        temperature=temp_pred,
        weather_type=weather_pred,
        confidence=confidence,
        is_cold_day=is_cold,
        location="Warsaw"
    )
    
    logger.info("\nForecast Narrative:")
    logger.info("-" * 40)
    print(narrative)
    
    return {
        'temperature': temp_pred,
        'weather_type': weather_pred,
        'weather_probs': weather_probs,
        'narrative': narrative
    }


def run_full_pipeline(args):
    """Run the complete MAS pipeline."""
    logger.info("🌤️ Weather Forecaster - Full Pipeline")
    logger.info("=" * 60)
    
    # Initialize configuration
    config = Config()
    config.seed = args.seed
    config.data.sequence_length = args.seq_length
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    
    # Phase 1: Data Retrieval
    data_path = run_data_retrieval(args, config)
    
    # Phase 2: Data Processing
    splits, loaders, data_agent = run_data_processing(data_path, config)
    
    baseline_agent = None
    transformer_agent = None
    
    # Phase 3: Baseline Training (optional)
    if not args.no_train_baseline:
        baseline_agent = run_baseline_training(splits, config)
    
    # Phase 4: Transformer Training (optional)
    if not args.no_train_transformer:
        transformer_agent = run_transformer_training(loaders, config, args)
    
    # Phase 5: Evaluation
    if baseline_agent and transformer_agent:
        comparison = run_evaluation(
            baseline_agent, transformer_agent, splits, loaders, config
        )
    
    # Phase 6: Prediction (use most recent data)
    if transformer_agent:
        predictions = run_prediction(transformer_agent, splits['test'], config)
    
    logger.info("=" * 60)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 60)
    
    logger.info("=" * 60)
    logger.info("✅ Pipeline completed successfully!")
    logger.info("=" * 60)


def run_ui():
    """Launch the Streamlit UI."""
    import subprocess
    logger.info("Launching Streamlit UI...")
    subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])


def main():
    """Main entry point."""
    args = get_args()
    
    try:
        if args.mode == "ui":
            run_ui()
        elif args.mode == "full":
            run_full_pipeline(args)
        elif args.mode == "train":
            # Training only mode
            args.no_train_baseline = False
            args.no_train_transformer = False
            run_full_pipeline(args)
        elif args.mode == "predict":
            # Prediction mode with existing models
            logger.info("Loading existing models for prediction...")
            config = Config()
            
            from src.agents.transformer_agent import TransformerAgent
            
            transformer_agent = TransformerAgent(config=config)
            model_path = config.models_dir / "transformer_model.pth"
            
            if model_path.exists():
                transformer_agent.load(model_path)
                
                # Get recent data
                from src.agents.data_retriever import DataRetriever
                retriever = DataRetriever()
                recent_df = retriever.fetch_current(
                    latitude=args.latitude,
                    longitude=args.longitude
                )
                
                if recent_df is not None:
                    logger.info("Making prediction with current weather data...")
                    # Note: Would need proper preprocessing here
                else:
                    logger.error("Failed to fetch current weather data")
            else:
                logger.error(f"Model not found at {model_path}. Run training first.")
                
        elif args.mode == "evaluate":
            logger.info("Evaluation mode - loading existing models...")
            config = Config()
            # Load and evaluate existing models
            # (Would need to implement model loading)
            
    except KeyboardInterrupt:
        logger.info("\n⚠️ Interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
