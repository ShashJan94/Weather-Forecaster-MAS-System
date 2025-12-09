"""
Baseline Agent
Implements traditional ML models for weather prediction as a comparison baseline.

This agent uses XGBoost and Random Forest to establish baseline performance
that the Transformer model should aim to beat.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier, XGBRegressor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BaselineResults:
    """Container for baseline model results."""
    model_name: str
    task: str  # 'regression' or 'classification'
    predictions: np.ndarray
    actuals: np.ndarray
    metrics: Dict[str, float]
    
    def summary(self) -> str:
        """Get a text summary of results."""
        lines = [f"\n{self.model_name} ({self.task}):"]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.4f}")
        return "\n".join(lines)


class BaselineAgent:
    """
    Agent for training and evaluating baseline ML models.
    
    Provides comparison baselines using:
    - XGBoost for both regression and classification
    - Random Forest as an alternative baseline
    """
    
    WEATHER_CLASSES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(
        self,
        use_xgboost: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the Baseline Agent.
        
        Args:
            use_xgboost: Whether to use XGBoost (True) or Random Forest (False)
            random_state: Random seed for reproducibility
        """
        self.use_xgboost = use_xgboost
        self.random_state = random_state
        
        # Initialize models
        if use_xgboost:
            self.regressor = XGBRegressor(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0
            )
            self.classifier = XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=random_state,
                n_jobs=-1,
                verbosity=0,
                eval_metric='mlogloss'
            )
            self.model_name = "XGBoost"
        else:
            self.regressor = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            self.classifier = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=random_state,
                n_jobs=-1
            )
            self.model_name = "RandomForest"
        
        self.is_trained = False
        self._train_results = None
        self._val_results = None
        self._test_results = None
    
    def train(
        self,
        X_train: np.ndarray,
        y_temp_train: np.ndarray,
        y_class_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_temp_val: Optional[np.ndarray] = None,
        y_class_val: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Train both regressor and classifier.
        
        Args:
            X_train: Training features
            y_temp_train: Temperature targets for training
            y_class_train: Class targets for training
            X_val: Validation features (optional)
            y_temp_val: Validation temperature targets
            y_class_val: Validation class targets
            
        Returns:
            Training results dictionary
        """
        logger.info(f"Training {self.model_name} baseline models...")
        logger.info(f"Training data shape: {X_train.shape}")
        
        # Train regressor
        logger.info("Training temperature regressor...")
        self.regressor.fit(X_train, y_temp_train)
        
        # Train classifier
        logger.info("Training weather classifier...")
        self.classifier.fit(X_train, y_class_train)
        
        self.is_trained = True
        
        # Evaluate on training data
        train_results = self.evaluate(X_train, y_temp_train, y_class_train, "train")
        self._train_results = train_results
        
        results = {"train": train_results}
        
        # Evaluate on validation data if provided
        if X_val is not None and y_temp_val is not None and y_class_val is not None:
            val_results = self.evaluate(X_val, y_temp_val, y_class_val, "val")
            self._val_results = val_results
            results["val"] = val_results
        
        logger.info(f"{self.model_name} training complete!")
        return results
    
    def predict(
        self,
        X: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Make predictions with trained models.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (temp_predictions, class_predictions, class_probabilities)
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet. Call train() first.")
        
        temp_pred = self.regressor.predict(X)
        class_pred = self.classifier.predict(X)
        class_probs = self.classifier.predict_proba(X)
        
        return temp_pred, class_pred, class_probs
    
    def evaluate(
        self,
        X: np.ndarray,
        y_temp: np.ndarray,
        y_class: np.ndarray,
        split_name: str = "test"
    ) -> Dict[str, BaselineResults]:
        """
        Evaluate models on given data.
        
        Args:
            X: Input features
            y_temp: True temperature values
            y_class: True class labels
            split_name: Name of the split for logging
            
        Returns:
            Dictionary with regression and classification results
        """
        temp_pred, class_pred, class_probs = self.predict(X)
        
        # Regression metrics
        reg_metrics = {
            "MAE": mean_absolute_error(y_temp, temp_pred),
            "RMSE": np.sqrt(mean_squared_error(y_temp, temp_pred)),
            "R2": r2_score(y_temp, temp_pred)
        }
        
        reg_results = BaselineResults(
            model_name=f"{self.model_name} Regressor",
            task="regression",
            predictions=temp_pred,
            actuals=y_temp,
            metrics=reg_metrics
        )
        
        # Classification metrics
        class_metrics = {
            "Accuracy": accuracy_score(y_class, class_pred),
            "F1_macro": f1_score(y_class, class_pred, average="macro"),
            "F1_weighted": f1_score(y_class, class_pred, average="weighted")
        }
        
        class_results = BaselineResults(
            model_name=f"{self.model_name} Classifier",
            task="classification",
            predictions=class_pred,
            actuals=y_class,
            metrics=class_metrics
        )
        
        logger.info(f"\n{split_name.upper()} Results:")
        logger.info(reg_results.summary())
        logger.info(class_results.summary())
        
        return {
            "regression": reg_results,
            "classification": class_results,
            "class_probabilities": class_probs
        }
    
    def test(
        self,
        X_test: np.ndarray,
        y_temp_test: np.ndarray,
        y_class_test: np.ndarray
    ) -> Dict[str, BaselineResults]:
        """
        Evaluate on test set.
        
        Args:
            X_test: Test features
            y_temp_test: Test temperature targets
            y_class_test: Test class targets
            
        Returns:
            Test results dictionary
        """
        self._test_results = self.evaluate(X_test, y_temp_test, y_class_test, "test")
        return self._test_results
    
    def get_feature_importance(self) -> Dict[str, np.ndarray]:
        """
        Get feature importance from trained models.
        
        Returns:
            Dictionary with feature importances for regressor and classifier
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        return {
            "regressor": self.regressor.feature_importances_,
            "classifier": self.classifier.feature_importances_
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
    
    def get_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray
    ) -> np.ndarray:
        """Get confusion matrix."""
        return confusion_matrix(y_true, y_pred)
    
    def save_models(self, path: Union[str, Path]) -> None:
        """
        Save trained models.
        
        Args:
            path: Directory to save models
        """
        if not self.is_trained:
            raise ValueError("Models not trained yet")
        
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.regressor, path / f"{self.model_name.lower()}_regressor.joblib")
        joblib.dump(self.classifier, path / f"{self.model_name.lower()}_classifier.joblib")
        
        logger.info(f"Saved {self.model_name} models to {path}")
    
    def load_models(self, path: Union[str, Path]) -> None:
        """
        Load trained models.
        
        Args:
            path: Directory containing saved models
        """
        path = Path(path)
        
        regressor_path = path / f"{self.model_name.lower()}_regressor.joblib"
        classifier_path = path / f"{self.model_name.lower()}_classifier.joblib"
        
        if regressor_path.exists() and classifier_path.exists():
            self.regressor = joblib.load(regressor_path)
            self.classifier = joblib.load(classifier_path)
            self.is_trained = True
            logger.info(f"Loaded {self.model_name} models from {path}")
        else:
            raise FileNotFoundError(f"Model files not found in {path}")
    
    def get_all_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        Get all collected metrics from training and evaluation.
        
        Returns:
            Dictionary of metrics for each split
        """
        metrics = {}
        
        for name, results in [
            ("train", self._train_results),
            ("val", self._val_results),
            ("test", self._test_results)
        ]:
            if results is not None:
                metrics[name] = {
                    **{f"temp_{k}": v for k, v in results["regression"].metrics.items()},
                    **{f"weather_{k}": v for k, v in results["classification"].metrics.items()}
                }
        
        return metrics
    
    def predict_single(
        self,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Make prediction for a single sample.
        
        Args:
            features: Feature vector for single sample (1D or 2D with shape (1, n_features))
            
        Returns:
            Dictionary with predictions
        """
        if features.ndim == 1:
            features = features.reshape(1, -1)
        
        temp_pred, class_pred, class_probs = self.predict(features)
        
        return {
            "temperature": float(temp_pred[0]),
            "weather_class": int(class_pred[0]),
            "weather_type": self.WEATHER_CLASSES[int(class_pred[0])],
            "class_probabilities": {
                cls: float(prob) 
                for cls, prob in zip(self.WEATHER_CLASSES, class_probs[0])
            }
        }


if __name__ == "__main__":
    # Test the baseline agent
    import sys
    from pathlib import Path
    
    # Add parent to path for imports
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
    splits, _ = data_agent.prepare_data(filepath=synthetic_path)
    aggregated = data_agent.get_aggregated_features(splits)
    
    # Train baseline
    agent = BaselineAgent(use_xgboost=True)
    
    X_train, y_temp_train, y_class_train = aggregated["train"]
    X_val, y_temp_val, y_class_val = aggregated["val"]
    X_test, y_temp_test, y_class_test = aggregated["test"]
    
    results = agent.train(
        X_train, y_temp_train, y_class_train,
        X_val, y_temp_val, y_class_val
    )
    
    # Test
    test_results = agent.test(X_test, y_temp_test, y_class_test)
    
    # Print classification report
    print("\nClassification Report:")
    print(agent.get_classification_report(
        test_results["classification"].actuals,
        test_results["classification"].predictions
    ))
    
    # Feature importance
    importance = agent.get_feature_importance()
    print("\nFeature Importance (Regressor):")
    print(importance["regressor"])
