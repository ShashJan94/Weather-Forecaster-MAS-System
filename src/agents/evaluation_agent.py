"""
Evaluation Agent
Compares and ensembles predictions from Baseline and Transformer agents.

This agent is responsible for:
- Comparing model performances
- Creating ensemble predictions
- Generating detailed performance reports
- Identifying model strengths and weaknesses
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ModelComparison:
    """Container for model comparison results."""
    baseline_metrics: Dict[str, float]
    transformer_metrics: Dict[str, float]
    ensemble_metrics: Dict[str, float]
    winner_regression: str
    winner_classification: str
    summary: str


@dataclass
class EnsemblePrediction:
    """Container for ensemble predictions."""
    temperature: float
    weather_class: int
    weather_type: str
    confidence: float
    baseline_contribution: float
    transformer_contribution: float
    method: str


class EvaluationAgent:
    """
    Agent for evaluating and comparing model predictions.
    
    Provides:
    - Model comparison metrics
    - Ensemble predictions
    - Performance analysis
    - Detailed reports
    """
    
    WEATHER_CLASSES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(
        self,
        ensemble_temp_weight_baseline: float = 0.5,
        ensemble_strategy: str = "average"  # "average", "weighted", "best"
    ):
        """
        Initialize the Evaluation Agent.
        
        Args:
            ensemble_temp_weight_baseline: Weight for baseline in temperature ensemble
            ensemble_strategy: Strategy for ensemble predictions
        """
        self.ensemble_temp_weight = ensemble_temp_weight_baseline
        self.ensemble_strategy = ensemble_strategy
        
        # Store results
        self._baseline_results = None
        self._transformer_results = None
        self._comparison = None
    
    def compare_models(
        self,
        baseline_temp_pred: np.ndarray,
        baseline_class_pred: np.ndarray,
        baseline_class_probs: np.ndarray,
        transformer_temp_pred: np.ndarray,
        transformer_class_pred: np.ndarray,
        transformer_class_probs: np.ndarray,
        temp_true: np.ndarray,
        class_true: np.ndarray
    ) -> ModelComparison:
        """
        Compare baseline and transformer predictions.
        
        Args:
            baseline_temp_pred: Baseline temperature predictions
            baseline_class_pred: Baseline class predictions
            baseline_class_probs: Baseline class probabilities
            transformer_temp_pred: Transformer temperature predictions
            transformer_class_pred: Transformer class predictions
            transformer_class_probs: Transformer class probabilities
            temp_true: True temperature values
            class_true: True class labels
            
        Returns:
            ModelComparison object with detailed comparison
        """
        logger.info("Comparing Baseline vs Transformer models...")
        
        # Compute baseline metrics
        baseline_metrics = self._compute_metrics(
            baseline_temp_pred, baseline_class_pred, temp_true, class_true
        )
        
        # Compute transformer metrics
        transformer_metrics = self._compute_metrics(
            transformer_temp_pred, transformer_class_pred, temp_true, class_true
        )
        
        # Compute ensemble predictions and metrics
        ensemble_temp, ensemble_class, ensemble_probs = self._create_ensemble(
            baseline_temp_pred, baseline_class_pred, baseline_class_probs,
            transformer_temp_pred, transformer_class_pred, transformer_class_probs,
            baseline_metrics, transformer_metrics
        )
        
        ensemble_metrics = self._compute_metrics(
            ensemble_temp, ensemble_class, temp_true, class_true
        )
        
        # Determine winners
        winner_reg = self._determine_winner(
            baseline_metrics["temp_MAE"],
            transformer_metrics["temp_MAE"],
            lower_is_better=True
        )
        
        winner_class = self._determine_winner(
            baseline_metrics["weather_Accuracy"],
            transformer_metrics["weather_Accuracy"],
            lower_is_better=False
        )
        
        # Generate summary
        summary = self._generate_summary(
            baseline_metrics, transformer_metrics, ensemble_metrics,
            winner_reg, winner_class, class_true, 
            baseline_class_pred, transformer_class_pred
        )
        
        comparison = ModelComparison(
            baseline_metrics=baseline_metrics,
            transformer_metrics=transformer_metrics,
            ensemble_metrics=ensemble_metrics,
            winner_regression=winner_reg,
            winner_classification=winner_class,
            summary=summary
        )
        
        self._comparison = comparison
        logger.info(f"\n{summary}")
        
        return comparison
    
    def _compute_metrics(
        self,
        temp_pred: np.ndarray,
        class_pred: np.ndarray,
        temp_true: np.ndarray,
        class_true: np.ndarray
    ) -> Dict[str, float]:
        """Compute all evaluation metrics."""
        return {
            "temp_MAE": mean_absolute_error(temp_true, temp_pred),
            "temp_RMSE": np.sqrt(mean_squared_error(temp_true, temp_pred)),
            "temp_R2": r2_score(temp_true, temp_pred),
            "weather_Accuracy": accuracy_score(class_true, class_pred),
            "weather_F1_macro": f1_score(class_true, class_pred, average="macro", zero_division=0),
            "weather_F1_weighted": f1_score(class_true, class_pred, average="weighted", zero_division=0)
        }
    
    def _create_ensemble(
        self,
        baseline_temp: np.ndarray,
        baseline_class: np.ndarray,
        baseline_probs: np.ndarray,
        transformer_temp: np.ndarray,
        transformer_class: np.ndarray,
        transformer_probs: np.ndarray,
        baseline_metrics: Dict[str, float],
        transformer_metrics: Dict[str, float]
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create ensemble predictions from both models.
        
        Returns:
            Tuple of (ensemble_temp, ensemble_class, ensemble_probs)
        """
        if self.ensemble_strategy == "average":
            # Simple average for temperature
            ensemble_temp = (baseline_temp + transformer_temp) / 2
            
            # Average probabilities for classification
            ensemble_probs = (baseline_probs + transformer_probs) / 2
            ensemble_class = ensemble_probs.argmax(axis=1)
            
        elif self.ensemble_strategy == "weighted":
            # Weighted by inverse MAE for temperature
            w_base = 1 / (baseline_metrics["temp_MAE"] + 1e-6)
            w_trans = 1 / (transformer_metrics["temp_MAE"] + 1e-6)
            w_sum = w_base + w_trans
            
            ensemble_temp = (w_base * baseline_temp + w_trans * transformer_temp) / w_sum
            
            # Weighted by accuracy for classification
            w_base_class = baseline_metrics["weather_Accuracy"]
            w_trans_class = transformer_metrics["weather_Accuracy"]
            w_sum_class = w_base_class + w_trans_class
            
            ensemble_probs = (w_base_class * baseline_probs + w_trans_class * transformer_probs) / w_sum_class
            ensemble_class = ensemble_probs.argmax(axis=1)
            
        else:  # "best" - use the better model for each task
            if baseline_metrics["temp_MAE"] < transformer_metrics["temp_MAE"]:
                ensemble_temp = baseline_temp
            else:
                ensemble_temp = transformer_temp
            
            if baseline_metrics["weather_Accuracy"] > transformer_metrics["weather_Accuracy"]:
                ensemble_probs = baseline_probs
                ensemble_class = baseline_class
            else:
                ensemble_probs = transformer_probs
                ensemble_class = transformer_class
        
        return ensemble_temp, ensemble_class, ensemble_probs
    
    def _determine_winner(
        self,
        baseline_value: float,
        transformer_value: float,
        lower_is_better: bool = True
    ) -> str:
        """Determine which model performs better."""
        if lower_is_better:
            if baseline_value < transformer_value:
                return "Baseline"
            elif transformer_value < baseline_value:
                return "Transformer"
            else:
                return "Tie"
        else:
            if baseline_value > transformer_value:
                return "Baseline"
            elif transformer_value > baseline_value:
                return "Transformer"
            else:
                return "Tie"
    
    def _generate_summary(
        self,
        baseline_metrics: Dict[str, float],
        transformer_metrics: Dict[str, float],
        ensemble_metrics: Dict[str, float],
        winner_reg: str,
        winner_class: str,
        class_true: np.ndarray,
        baseline_class_pred: np.ndarray,
        transformer_class_pred: np.ndarray
    ) -> str:
        """Generate a human-readable summary of comparison."""
        lines = [
            "=" * 60,
            "MODEL COMPARISON SUMMARY",
            "=" * 60,
            "",
            "TEMPERATURE PREDICTION (Regression):",
            f"  Baseline MAE:     {baseline_metrics['temp_MAE']:.2f}°C",
            f"  Transformer MAE:  {transformer_metrics['temp_MAE']:.2f}°C",
            f"  Ensemble MAE:     {ensemble_metrics['temp_MAE']:.2f}°C",
            f"  Winner: {winner_reg}",
            "",
            "WEATHER CLASSIFICATION:",
            f"  Baseline Accuracy:     {baseline_metrics['weather_Accuracy']:.2%}",
            f"  Transformer Accuracy:  {transformer_metrics['weather_Accuracy']:.2%}",
            f"  Ensemble Accuracy:     {ensemble_metrics['weather_Accuracy']:.2%}",
            f"  Winner: {winner_class}",
            "",
            "PER-CLASS ANALYSIS:",
        ]
        
        # Per-class accuracy
        for i, cls in enumerate(self.WEATHER_CLASSES):
            mask = class_true == i
            if mask.sum() > 0:
                base_acc = (baseline_class_pred[mask] == i).mean()
                trans_acc = (transformer_class_pred[mask] == i).mean()
                better = "Baseline" if base_acc > trans_acc else "Transformer" if trans_acc > base_acc else "Tie"
                lines.append(f"  {cls.capitalize()}: Baseline {base_acc:.1%}, Transformer {trans_acc:.1%} -> {better}")
        
        lines.extend([
            "",
            "RECOMMENDATIONS:",
        ])
        
        if winner_reg == "Transformer" and winner_class == "Transformer":
            lines.append("  -> Transformer outperforms on both tasks. Use Transformer model.")
        elif winner_reg == "Baseline" and winner_class == "Baseline":
            lines.append("  -> Baseline outperforms on both tasks. Consider using Baseline for efficiency.")
        else:
            lines.append("  -> Models have different strengths. Consider ensemble approach.")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def ensemble_predict(
        self,
        baseline_prediction: Dict[str, Any],
        transformer_prediction: Dict[str, Any]
    ) -> EnsemblePrediction:
        """
        Create an ensemble prediction from both models.
        
        Args:
            baseline_prediction: Prediction dict from baseline agent
            transformer_prediction: Prediction dict from transformer agent
            
        Returns:
            EnsemblePrediction object
        """
        # Temperature ensemble
        base_temp = baseline_prediction["temperature"]
        trans_temp = transformer_prediction["temperature"]
        
        if self.ensemble_strategy == "average":
            ensemble_temp = (base_temp + trans_temp) / 2
            base_contrib = 0.5
            trans_contrib = 0.5
        elif self.ensemble_strategy == "weighted":
            # If we have comparison results, use performance-based weights
            if self._comparison is not None:
                base_mae = self._comparison.baseline_metrics["temp_MAE"]
                trans_mae = self._comparison.transformer_metrics["temp_MAE"]
                w_base = 1 / (base_mae + 1e-6)
                w_trans = 1 / (trans_mae + 1e-6)
                w_sum = w_base + w_trans
                base_contrib = w_base / w_sum
                trans_contrib = w_trans / w_sum
            else:
                base_contrib = 0.5
                trans_contrib = 0.5
            ensemble_temp = base_contrib * base_temp + trans_contrib * trans_temp
        else:  # best
            if self._comparison is not None:
                if self._comparison.winner_regression == "Baseline":
                    ensemble_temp = base_temp
                    base_contrib = 1.0
                    trans_contrib = 0.0
                else:
                    ensemble_temp = trans_temp
                    base_contrib = 0.0
                    trans_contrib = 1.0
            else:
                ensemble_temp = (base_temp + trans_temp) / 2
                base_contrib = 0.5
                trans_contrib = 0.5
        
        # Classification ensemble (probability averaging)
        base_probs = baseline_prediction["class_probabilities"]
        trans_probs = transformer_prediction["class_probabilities"]
        
        ensemble_probs = {}
        for cls in self.WEATHER_CLASSES:
            ensemble_probs[cls] = (base_probs.get(cls, 0) + trans_probs.get(cls, 0)) / 2
        
        # Get predicted class
        best_class = max(ensemble_probs.keys(), key=lambda k: ensemble_probs[k])
        class_idx = self.WEATHER_CLASSES.index(best_class)
        confidence = ensemble_probs[best_class]
        
        return EnsemblePrediction(
            temperature=ensemble_temp,
            weather_class=class_idx,
            weather_type=best_class,
            confidence=confidence,
            baseline_contribution=base_contrib,
            transformer_contribution=trans_contrib,
            method=self.ensemble_strategy
        )
    
    def get_detailed_report(
        self,
        y_true_temp: np.ndarray,
        y_pred_temp_baseline: np.ndarray,
        y_pred_temp_transformer: np.ndarray,
        y_true_class: np.ndarray,
        y_pred_class_baseline: np.ndarray,
        y_pred_class_transformer: np.ndarray
    ) -> str:
        """
        Generate a detailed comparison report.
        
        Returns:
            Detailed report as string
        """
        lines = [
            "\n" + "=" * 70,
            "DETAILED MODEL EVALUATION REPORT",
            "=" * 70,
            "",
            "1. TEMPERATURE PREDICTION ANALYSIS",
            "-" * 40,
        ]
        
        for name, pred in [("Baseline", y_pred_temp_baseline), 
                           ("Transformer", y_pred_temp_transformer)]:
            mae = mean_absolute_error(y_true_temp, pred)
            rmse = np.sqrt(mean_squared_error(y_true_temp, pred))
            r2 = r2_score(y_true_temp, pred)
            
            errors = y_true_temp - pred
            lines.extend([
                f"\n{name}:",
                f"  MAE:  {mae:.3f}°C",
                f"  RMSE: {rmse:.3f}°C",
                f"  R²:   {r2:.3f}",
                f"  Mean Error: {errors.mean():.3f}°C",
                f"  Std Error:  {errors.std():.3f}°C",
            ])
        
        lines.extend([
            "",
            "2. WEATHER CLASSIFICATION ANALYSIS",
            "-" * 40,
        ])
        
        for name, pred in [("Baseline", y_pred_class_baseline),
                           ("Transformer", y_pred_class_transformer)]:
            lines.append(f"\n{name} Classification Report:")
            report = classification_report(
                y_true_class, pred,
                target_names=self.WEATHER_CLASSES,
                zero_division=0
            )
            lines.append(report)
        
        lines.extend([
            "",
            "3. ERROR ANALYSIS",
            "-" * 40,
        ])
        
        # Find cases where models disagree
        disagree_mask = y_pred_class_baseline != y_pred_class_transformer
        disagree_count = disagree_mask.sum()
        
        lines.append(f"\nModels disagree on {disagree_count} samples ({disagree_count/len(y_true_class)*100:.1f}%)")
        
        # When they disagree, who is right?
        if disagree_count > 0:
            base_correct = (y_pred_class_baseline[disagree_mask] == y_true_class[disagree_mask]).sum()
            trans_correct = (y_pred_class_transformer[disagree_mask] == y_true_class[disagree_mask]).sum()
            lines.append(f"  When disagreeing, Baseline correct: {base_correct} times")
            lines.append(f"  When disagreeing, Transformer correct: {trans_correct} times")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def get_per_class_metrics(
        self,
        y_true: np.ndarray,
        baseline_pred: np.ndarray,
        transformer_pred: np.ndarray
    ) -> pd.DataFrame:
        """
        Get per-class metrics for both models.
        
        Returns:
            DataFrame with per-class comparison
        """
        data = []
        
        for i, cls in enumerate(self.WEATHER_CLASSES):
            mask = y_true == i
            n_samples = mask.sum()
            
            if n_samples > 0:
                base_acc = (baseline_pred[mask] == i).mean()
                trans_acc = (transformer_pred[mask] == i).mean()
                
                data.append({
                    "Class": cls,
                    "Samples": n_samples,
                    "Baseline_Accuracy": base_acc,
                    "Transformer_Accuracy": trans_acc,
                    "Difference": trans_acc - base_acc,
                    "Better": "Transformer" if trans_acc > base_acc else "Baseline" if base_acc > trans_acc else "Tie"
                })
        
        return pd.DataFrame(data)
    
    @property
    def comparison_results(self) -> Optional[ModelComparison]:
        """Get stored comparison results."""
        return self._comparison


if __name__ == "__main__":
    # Test evaluation agent
    np.random.seed(42)
    
    # Simulate predictions
    n_samples = 100
    temp_true = np.random.randn(n_samples) * 10 + 15
    class_true = np.random.randint(0, 4, n_samples)
    
    # Baseline predictions (slightly noisy)
    baseline_temp = temp_true + np.random.randn(n_samples) * 2
    baseline_class = class_true.copy()
    baseline_class[np.random.rand(n_samples) < 0.15] = np.random.randint(0, 4, (np.random.rand(n_samples) < 0.15).sum())
    baseline_probs = np.eye(4)[baseline_class] * 0.7 + 0.075
    
    # Transformer predictions (less noisy)
    transformer_temp = temp_true + np.random.randn(n_samples) * 1.5
    transformer_class = class_true.copy()
    transformer_class[np.random.rand(n_samples) < 0.1] = np.random.randint(0, 4, (np.random.rand(n_samples) < 0.1).sum())
    transformer_probs = np.eye(4)[transformer_class] * 0.8 + 0.05
    
    # Evaluate
    agent = EvaluationAgent(ensemble_strategy="weighted")
    
    comparison = agent.compare_models(
        baseline_temp, baseline_class, baseline_probs,
        transformer_temp, transformer_class, transformer_probs,
        temp_true, class_true
    )
    
    print("\nPer-class metrics:")
    print(agent.get_per_class_metrics(class_true, baseline_class, transformer_class))
