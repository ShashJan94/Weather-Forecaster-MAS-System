"""
Transformer Agent
Handles training, evaluation, and inference with the Tiny Weather Transformer.

This agent is responsible for:
- Building and configuring the Transformer model
- Training with early stopping
- Evaluation and metrics computation
- Making predictions
"""

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report
)
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from tqdm import tqdm
import logging

from ..models.transformer_model import (
    TinyWeatherTransformer,
    TransformerModelConfig,
    WeatherLoss
)
from ..utils.helpers import set_seed, get_device, save_model, load_model
from .data_agent import DataSplit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class TrainingHistory:
    """Container for training history."""
    train_losses: List[float]
    val_losses: List[float]
    train_metrics: List[Dict[str, float]]
    val_metrics: List[Dict[str, float]]
    best_epoch: int
    best_val_loss: float


@dataclass 
class TransformerResults:
    """Container for Transformer evaluation results."""
    temp_predictions: np.ndarray
    temp_actuals: np.ndarray
    class_predictions: np.ndarray
    class_actuals: np.ndarray
    class_probabilities: np.ndarray
    metrics: Dict[str, float]
    
    def summary(self) -> str:
        """Get text summary of results."""
        lines = ["\nTransformer Results:"]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class TransformerAgent:
    """
    Agent for training and inference with the Tiny Weather Transformer.
    """
    
    WEATHER_CLASSES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(
        self,
        num_features: int = 4,
        d_model: int = 64,
        num_heads: int = 2,
        num_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_seq_len: int = 14,
        num_classes: int = 4,
        device: Optional[torch.device] = None,
        seed: int = 42
    ):
        """
        Initialize the Transformer Agent.
        
        Args:
            num_features: Number of input features
            d_model: Model dimension
            num_heads: Number of attention heads
            num_layers: Number of encoder layers
            d_ff: Feed-forward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
            num_classes: Number of output classes
            device: Device for training/inference
            seed: Random seed
        """
        set_seed(seed)
        self.device = device or get_device()
        logger.info(f"Using device: {self.device}")
        
        # Model configuration
        self.config = TransformerModelConfig(
            num_features=num_features,
            d_model=d_model,
            num_heads=num_heads,
            num_layers=num_layers,
            d_ff=d_ff,
            dropout=dropout,
            max_seq_len=max_seq_len,
            num_classes=num_classes
        )
        
        # Build model
        self.model = TinyWeatherTransformer(self.config).to(self.device)
        
        # Log model size
        params = self.model.get_num_parameters()
        logger.info(f"Model parameters: {params['total']:,} total, {params['trainable']:,} trainable")
        
        # Training state
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.history = None
        self.is_trained = False
        
        # Results storage
        self._train_results = None
        self._val_results = None
        self._test_results = None
    
    def build_training_components(
        self,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        lambda_temp: float = 1.0,
        lambda_weather: float = 1.0,
        lambda_cold: float = 0.5,
        num_epochs: int = 30,
        class_weights: Optional[torch.Tensor] = None
    ):
        """
        Build optimizer, scheduler, and loss function.
        
        Args:
            learning_rate: Initial learning rate
            weight_decay: L2 regularization weight
            lambda_temp: Weight for temperature loss
            lambda_weather: Weight for weather classification loss
            lambda_cold: Weight for cold day classification loss
            num_epochs: Number of training epochs (for scheduler)
            class_weights: Optional class weights for imbalanced data
        """
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=num_epochs,
            eta_min=1e-6
        )
        
        if class_weights is not None:
            class_weights = class_weights.to(self.device)
        
        self.criterion = WeatherLoss(
            lambda_temp=lambda_temp,
            lambda_weather=lambda_weather,
            lambda_cold=lambda_cold,
            class_weights=class_weights
        )
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 30,
        learning_rate: float = 1e-3,
        early_stopping_patience: int = 5,
        save_path: Optional[Path] = None
    ) -> TrainingHistory:
        """
        Train the Transformer model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            num_epochs: Maximum number of epochs
            learning_rate: Learning rate
            early_stopping_patience: Epochs to wait before early stopping
            save_path: Path to save best model
            
        Returns:
            TrainingHistory object
        """
        logger.info("Starting Transformer training...")
        
        # Build training components if not already done
        if self.optimizer is None:
            self.build_training_components(
                learning_rate=learning_rate,
                num_epochs=num_epochs
            )
        
        # Training history
        history = TrainingHistory(
            train_losses=[],
            val_losses=[],
            train_metrics=[],
            val_metrics=[],
            best_epoch=0,
            best_val_loss=float('inf')
        )
        
        # Early stopping
        patience_counter = 0
        best_model_state = None
        
        for epoch in range(num_epochs):
            # Training phase
            train_loss, train_metrics = self._train_epoch(train_loader)
            history.train_losses.append(train_loss)
            history.train_metrics.append(train_metrics)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            history.val_losses.append(val_loss)
            history.val_metrics.append(val_metrics)
            
            # Update scheduler
            self.scheduler.step()
            
            # Logging
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_metrics.get('weather_accuracy', 0):.4f} | "
                f"Val MAE: {val_metrics.get('temp_mae', 0):.4f}"
            )
            
            # Check for improvement
            if val_loss < history.best_val_loss:
                history.best_val_loss = val_loss
                history.best_epoch = epoch
                best_model_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                
                if save_path:
                    save_model(self.model, save_path, self.optimizer, epoch, val_metrics)
            else:
                patience_counter += 1
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
        
        self.history = history
        self.is_trained = True
        
        logger.info(f"Training complete! Best epoch: {history.best_epoch+1}, Best val loss: {history.best_val_loss:.4f}")
        return history
    
    def _train_epoch(self, train_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run one training epoch."""
        self.model.train()
        total_loss = 0.0
        all_preds = {"temp": [], "weather": [], "cold": []}
        all_targets = {"temp": [], "weather": [], "cold": []}
        
        for batch in train_loader:
            # Move to device
            sequences = batch["sequence"].to(self.device)
            targets = {
                "temp": batch["temp"].to(self.device),
                "weather_class": batch["weather_class"].to(self.device),
                "cold_day": batch["cold_day"].to(self.device)
            }
            
            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(sequences)
            
            # Compute loss
            loss, _ = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Collect predictions
            all_preds["temp"].append(outputs["temp_pred"].detach().cpu().numpy())
            all_preds["weather"].append(outputs["weather_logits"].argmax(dim=1).cpu().numpy())
            all_targets["temp"].append(targets["temp"].cpu().numpy())
            all_targets["weather"].append(targets["weather_class"].cpu().numpy())
        
        # Compute metrics
        avg_loss = total_loss / len(train_loader)
        metrics = self._compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _validate_epoch(self, val_loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        """Run one validation epoch."""
        self.model.eval()
        total_loss = 0.0
        all_preds = {"temp": [], "weather": [], "cold": []}
        all_targets = {"temp": [], "weather": [], "cold": []}
        
        with torch.no_grad():
            for batch in val_loader:
                sequences = batch["sequence"].to(self.device)
                targets = {
                    "temp": batch["temp"].to(self.device),
                    "weather_class": batch["weather_class"].to(self.device),
                    "cold_day": batch["cold_day"].to(self.device)
                }
                
                outputs = self.model(sequences)
                loss, _ = self.criterion(outputs, targets)
                
                total_loss += loss.item()
                
                all_preds["temp"].append(outputs["temp_pred"].cpu().numpy())
                all_preds["weather"].append(outputs["weather_logits"].argmax(dim=1).cpu().numpy())
                all_targets["temp"].append(targets["temp"].cpu().numpy())
                all_targets["weather"].append(targets["weather_class"].cpu().numpy())
        
        avg_loss = total_loss / len(val_loader)
        metrics = self._compute_metrics(all_preds, all_targets)
        
        return avg_loss, metrics
    
    def _compute_metrics(
        self,
        preds: Dict[str, List[np.ndarray]],
        targets: Dict[str, List[np.ndarray]]
    ) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Concatenate predictions
        temp_pred = np.concatenate(preds["temp"]).ravel()
        temp_true = np.concatenate(targets["temp"]).ravel()
        weather_pred = np.concatenate(preds["weather"])
        weather_true = np.concatenate(targets["weather"])
        
        return {
            "temp_mae": mean_absolute_error(temp_true, temp_pred),
            "temp_rmse": np.sqrt(mean_squared_error(temp_true, temp_pred)),
            "temp_r2": r2_score(temp_true, temp_pred),
            "weather_accuracy": accuracy_score(weather_true, weather_pred),
            "weather_f1_macro": f1_score(weather_true, weather_pred, average="macro", zero_division=0),
            "weather_f1_weighted": f1_score(weather_true, weather_pred, average="weighted", zero_division=0)
        }
    
    def evaluate(
        self,
        data_loader: DataLoader,
        split_name: str = "test"
    ) -> TransformerResults:
        """
        Evaluate the model on a dataset.
        
        Args:
            data_loader: Data loader for evaluation
            split_name: Name of the split for logging
            
        Returns:
            TransformerResults object
        """
        self.model.eval()
        all_temp_pred = []
        all_temp_true = []
        all_class_pred = []
        all_class_true = []
        all_class_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                sequences = batch["sequence"].to(self.device)
                
                outputs = self.model.predict(sequences, return_probs=True)
                
                all_temp_pred.append(outputs["temp_pred"].cpu().numpy())
                all_temp_true.append(batch["temp"].numpy())
                all_class_pred.append(outputs["weather_pred"].cpu().numpy())
                all_class_true.append(batch["weather_class"].numpy())
                all_class_probs.append(outputs["weather_probs"].cpu().numpy())
        
        # Concatenate results
        temp_pred = np.concatenate(all_temp_pred).ravel()
        temp_true = np.concatenate(all_temp_true).ravel()
        class_pred = np.concatenate(all_class_pred)
        class_true = np.concatenate(all_class_true)
        class_probs = np.concatenate(all_class_probs)
        
        # Compute metrics
        metrics = {
            "temp_MAE": mean_absolute_error(temp_true, temp_pred),
            "temp_RMSE": np.sqrt(mean_squared_error(temp_true, temp_pred)),
            "temp_R2": r2_score(temp_true, temp_pred),
            "weather_Accuracy": accuracy_score(class_true, class_pred),
            "weather_F1_macro": f1_score(class_true, class_pred, average="macro", zero_division=0),
            "weather_F1_weighted": f1_score(class_true, class_pred, average="weighted", zero_division=0)
        }
        
        results = TransformerResults(
            temp_predictions=temp_pred,
            temp_actuals=temp_true,
            class_predictions=class_pred,
            class_actuals=class_true,
            class_probabilities=class_probs,
            metrics=metrics
        )
        
        logger.info(f"\n{split_name.upper()} Results:{results.summary()}")
        
        # Store results
        if split_name == "train":
            self._train_results = results
        elif split_name == "val":
            self._val_results = results
        elif split_name == "test":
            self._test_results = results
        
        return results
    
    def predict(
        self,
        sequence: torch.Tensor
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sequence.
        
        Args:
            sequence: Input tensor of shape (1, seq_len, num_features) or (seq_len, num_features)
            
        Returns:
            Dictionary with predictions
        """
        self.model.eval()
        
        # Add batch dimension if needed
        if sequence.dim() == 2:
            sequence = sequence.unsqueeze(0)
        
        sequence = sequence.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.predict(sequence, return_probs=True)
        
        # Extract single prediction
        weather_probs = outputs["weather_probs"][0].cpu().numpy()
        
        return {
            "temperature": float(outputs["temp_pred"][0, 0].cpu()),
            "weather_class": int(outputs["weather_pred"][0].cpu()),
            "weather_type": self.WEATHER_CLASSES[int(outputs["weather_pred"][0].cpu())],
            "class_probabilities": {
                cls: float(prob)
                for cls, prob in zip(self.WEATHER_CLASSES, weather_probs)
            },
            "is_cold_day": bool(outputs["cold_day_pred"][0].cpu())
        }
    
    def get_classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> str:
        """Get detailed classification report."""
        return classification_report(
            y_true, y_pred,
            target_names=self.WEATHER_CLASSES,
            zero_division=0
        )
    
    def save(self, path: Union[str, Path]) -> None:
        """Save model and training state."""
        path = Path(path)
        save_model(
            self.model,
            path,
            optimizer=self.optimizer,
            metrics=self._test_results.metrics if self._test_results else None
        )
        logger.info(f"Saved Transformer model to {path}")
    
    def load(self, path: Union[str, Path]) -> None:
        """Load saved model."""
        path = Path(path)
        result = load_model(self.model, path, self.device)
        self.is_trained = True
        logger.info(f"Loaded Transformer model from {path}")
        return result
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """Get all collected metrics."""
        metrics = {}
        for name, results in [
            ("train", self._train_results),
            ("val", self._val_results),
            ("test", self._test_results)
        ]:
            if results is not None:
                metrics[name] = results.metrics
        return metrics
    
    def get_training_history(self) -> Optional[TrainingHistory]:
        """Get training history."""
        return self.history


if __name__ == "__main__":
    # Test the transformer agent
    import sys
    from pathlib import Path
    
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    from src.agents.data_agent import DataAgent
    from src.agents.data_retriever import generate_synthetic_data
    
    # Generate synthetic data
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    data_dir.mkdir(parents=True, exist_ok=True)
    synthetic_path = data_dir / "synthetic_weather.csv"
    
    if not synthetic_path.exists():
        generate_synthetic_data(save_path=synthetic_path)
    
    # Prepare data
    data_agent = DataAgent(sequence_length=7)
    splits, loaders = data_agent.prepare_data(filepath=synthetic_path, batch_size=32)
    
    # Create and train transformer agent
    agent = TransformerAgent(
        num_features=data_agent.num_features,
        d_model=64,
        num_heads=2,
        num_layers=2
    )
    
    # Train (just a few epochs for testing)
    history = agent.train(
        loaders["train"],
        loaders["val"],
        num_epochs=5,
        early_stopping_patience=3
    )
    
    # Evaluate on test set
    test_results = agent.evaluate(loaders["test"], "test")
    
    print("\nClassification Report:")
    print(agent.get_classification_report(
        test_results.class_actuals,
        test_results.class_predictions
    ))
