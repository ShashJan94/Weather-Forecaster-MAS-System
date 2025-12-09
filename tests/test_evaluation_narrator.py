"""
Tests for the Evaluation Agent and Narrator Agent.
"""

import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.evaluation_agent import EvaluationAgent, ModelComparison, EnsemblePrediction
from src.agents.narrator_agent import NarratorAgent


class TestEvaluationAgent:
    """Test cases for EvaluationAgent class."""
    
    @pytest.fixture
    def sample_predictions(self):
        """Create sample predictions for testing."""
        np.random.seed(42)
        n_samples = 50
        
        temp_true = np.random.randn(n_samples) * 10 + 15
        class_true = np.random.randint(0, 4, n_samples)
        
        # Baseline predictions (more noisy)
        baseline_temp = temp_true + np.random.randn(n_samples) * 3
        baseline_class = class_true.copy()
        noise_mask = np.random.rand(n_samples) < 0.2
        baseline_class[noise_mask] = np.random.randint(0, 4, noise_mask.sum())
        baseline_probs = np.eye(4)[baseline_class] * 0.7 + 0.075
        
        # Transformer predictions (less noisy)
        transformer_temp = temp_true + np.random.randn(n_samples) * 2
        transformer_class = class_true.copy()
        noise_mask = np.random.rand(n_samples) < 0.1
        transformer_class[noise_mask] = np.random.randint(0, 4, noise_mask.sum())
        transformer_probs = np.eye(4)[transformer_class] * 0.8 + 0.05
        
        return {
            "temp_true": temp_true,
            "class_true": class_true,
            "baseline_temp": baseline_temp,
            "baseline_class": baseline_class,
            "baseline_probs": baseline_probs,
            "transformer_temp": transformer_temp,
            "transformer_class": transformer_class,
            "transformer_probs": transformer_probs
        }
    
    def test_initialization(self):
        """Test EvaluationAgent initialization."""
        agent = EvaluationAgent(ensemble_strategy="weighted")
        
        assert agent.ensemble_strategy == "weighted"
        assert agent._comparison is None
    
    def test_compare_models(self, sample_predictions):
        """Test model comparison."""
        agent = EvaluationAgent()
        
        comparison = agent.compare_models(
            sample_predictions["baseline_temp"],
            sample_predictions["baseline_class"],
            sample_predictions["baseline_probs"],
            sample_predictions["transformer_temp"],
            sample_predictions["transformer_class"],
            sample_predictions["transformer_probs"],
            sample_predictions["temp_true"],
            sample_predictions["class_true"]
        )
        
        assert isinstance(comparison, ModelComparison)
        assert "temp_MAE" in comparison.baseline_metrics
        assert "weather_Accuracy" in comparison.transformer_metrics
        assert comparison.winner_regression in ["Baseline", "Transformer", "Tie"]
        assert comparison.winner_classification in ["Baseline", "Transformer", "Tie"]
        assert len(comparison.summary) > 0
    
    def test_ensemble_predict_average(self, sample_predictions):
        """Test ensemble prediction with average strategy."""
        agent = EvaluationAgent(ensemble_strategy="average")
        
        baseline_pred = {
            "temperature": 15.0,
            "weather_type": "cloudy",
            "class_probabilities": {"sunny": 0.1, "cloudy": 0.6, "rainy": 0.2, "snowy": 0.1}
        }
        transformer_pred = {
            "temperature": 13.0,
            "weather_type": "rainy",
            "class_probabilities": {"sunny": 0.05, "cloudy": 0.25, "rainy": 0.6, "snowy": 0.1}
        }
        
        ensemble = agent.ensemble_predict(baseline_pred, transformer_pred)
        
        assert isinstance(ensemble, EnsemblePrediction)
        assert ensemble.temperature == 14.0  # Average
        assert ensemble.method == "average"
        assert ensemble.baseline_contribution == 0.5
        assert ensemble.transformer_contribution == 0.5
    
    def test_ensemble_predict_weighted(self, sample_predictions):
        """Test ensemble prediction with weighted strategy after comparison."""
        agent = EvaluationAgent(ensemble_strategy="weighted")
        
        # First do comparison to establish weights
        agent.compare_models(
            sample_predictions["baseline_temp"],
            sample_predictions["baseline_class"],
            sample_predictions["baseline_probs"],
            sample_predictions["transformer_temp"],
            sample_predictions["transformer_class"],
            sample_predictions["transformer_probs"],
            sample_predictions["temp_true"],
            sample_predictions["class_true"]
        )
        
        baseline_pred = {
            "temperature": 15.0,
            "weather_type": "cloudy",
            "class_probabilities": {"sunny": 0.1, "cloudy": 0.6, "rainy": 0.2, "snowy": 0.1}
        }
        transformer_pred = {
            "temperature": 13.0,
            "weather_type": "rainy",
            "class_probabilities": {"sunny": 0.05, "cloudy": 0.25, "rainy": 0.6, "snowy": 0.1}
        }
        
        ensemble = agent.ensemble_predict(baseline_pred, transformer_pred)
        
        assert isinstance(ensemble, EnsemblePrediction)
        assert ensemble.method == "weighted"
        # Weights should be based on performance
        assert ensemble.baseline_contribution + ensemble.transformer_contribution == pytest.approx(1.0)
    
    def test_detailed_report(self, sample_predictions):
        """Test detailed report generation."""
        agent = EvaluationAgent()
        
        report = agent.get_detailed_report(
            sample_predictions["temp_true"],
            sample_predictions["baseline_temp"],
            sample_predictions["transformer_temp"],
            sample_predictions["class_true"],
            sample_predictions["baseline_class"],
            sample_predictions["transformer_class"]
        )
        
        assert isinstance(report, str)
        assert "TEMPERATURE" in report
        assert "CLASSIFICATION" in report
        assert "Baseline" in report
        assert "Transformer" in report
    
    def test_per_class_metrics(self, sample_predictions):
        """Test per-class metrics calculation."""
        agent = EvaluationAgent()
        
        df = agent.get_per_class_metrics(
            sample_predictions["class_true"],
            sample_predictions["baseline_class"],
            sample_predictions["transformer_class"]
        )
        
        assert "Class" in df.columns
        assert "Baseline_Accuracy" in df.columns
        assert "Transformer_Accuracy" in df.columns
        assert "Better" in df.columns


class TestNarratorAgent:
    """Test cases for NarratorAgent class."""
    
    def test_initialization(self):
        """Test NarratorAgent initialization."""
        agent = NarratorAgent(use_emoji=True)
        
        assert agent.use_emoji == True
        assert "sunny" in agent.WEATHER_INFO
        assert "rainy" in agent.WEATHER_INFO
    
    def test_generate_forecast(self):
        """Test forecast generation."""
        agent = NarratorAgent(use_emoji=True)
        
        forecast = agent.generate_forecast(
            temperature=15.5,
            weather_type="sunny",
            confidence=0.85,
            is_cold_day=False,
            location="Warsaw",
            target_date="Tomorrow"
        )
        
        assert "headline" in forecast
        assert "description" in forecast
        assert "recommendation" in forecast
        assert "full_narrative" in forecast
        assert "icon" in forecast
        assert "☀️" in forecast["icon"]
    
    def test_generate_forecast_cold_day(self):
        """Test forecast generation for cold day."""
        agent = NarratorAgent()
        
        forecast = agent.generate_forecast(
            temperature=2.0,
            weather_type="snowy",
            confidence=0.9,
            is_cold_day=True,
            location="Moscow"
        )
        
        assert "cold" in forecast["full_narrative"].lower() or "bundle" in forecast["full_narrative"].lower()
    
    def test_generate_forecast_low_confidence(self):
        """Test forecast generation with low confidence."""
        agent = NarratorAgent()
        
        forecast = agent.generate_forecast(
            temperature=12.0,
            weather_type="cloudy",
            confidence=0.4,
            is_cold_day=False,
            location="London"
        )
        
        # Should include confidence warning
        assert "uncertain" in forecast["confidence_note"].lower() or "⚠️" in forecast["confidence_note"]
    
    def test_generate_comparison_narrative(self):
        """Test comparison narrative generation."""
        agent = NarratorAgent()
        
        baseline_pred = {"temperature": 14.0, "weather_type": "cloudy"}
        transformer_pred = {"temperature": 15.0, "weather_type": "sunny"}
        ensemble_pred = {"temperature": 14.5, "weather_type": "cloudy", "confidence": 0.75}
        
        narrative = agent.generate_comparison_narrative(
            baseline_pred, transformer_pred, ensemble_pred
        )
        
        assert "Baseline" in narrative
        assert "Transformer" in narrative
        assert "14" in narrative
        assert "15" in narrative
    
    def test_generate_comparison_narrative_agreement(self):
        """Test comparison narrative when models agree."""
        agent = NarratorAgent()
        
        baseline_pred = {"temperature": 14.0, "weather_type": "sunny"}
        transformer_pred = {"temperature": 14.2, "weather_type": "sunny"}
        
        narrative = agent.generate_comparison_narrative(baseline_pred, transformer_pred)
        
        assert "agree" in narrative.lower()
    
    def test_generate_weekly_summary(self):
        """Test weekly summary generation."""
        agent = NarratorAgent()
        
        daily_forecasts = [
            {"temperature": 12, "weather_type": "rainy"},
            {"temperature": 14, "weather_type": "cloudy"},
            {"temperature": 16, "weather_type": "sunny"},
            {"temperature": 18, "weather_type": "sunny"},
            {"temperature": 15, "weather_type": "cloudy"},
            {"temperature": 13, "weather_type": "rainy"},
            {"temperature": 11, "weather_type": "rainy"},
        ]
        
        summary = agent.generate_weekly_summary(daily_forecasts)
        
        assert "Weekly" in summary
        assert "Temperature Range" in summary
        assert "11" in summary  # min temp
        assert "18" in summary  # max temp
    
    def test_generate_weekly_summary_empty(self):
        """Test weekly summary with no data."""
        agent = NarratorAgent()
        
        summary = agent.generate_weekly_summary([])
        
        assert "No forecast data" in summary
    
    def test_all_weather_types_covered(self):
        """Test that all weather types have proper icons and descriptions."""
        agent = NarratorAgent()
        
        for weather_type in ["sunny", "cloudy", "rainy", "snowy"]:
            info = agent.WEATHER_INFO.get(weather_type)
            
            assert info is not None
            assert "icon" in info
            assert "descriptions" in info
            assert "tips" in info
            assert len(info["descriptions"]) > 0
            assert len(info["tips"]) > 0
    
    def test_temperature_descriptors(self):
        """Test temperature descriptors for various temperatures."""
        agent = NarratorAgent()
        
        # Test various temperatures
        test_cases = [
            (-15, "frigid"),
            (-5, "freezing"),
            (2, "cold"),
            (8, "cool"),
            (15, "mild"),
            (22, "warm"),
            (28, "hot"),
            (35, "hot")
        ]
        
        for temp, expected_category in test_cases:
            desc = agent._get_temp_descriptor(temp)
            assert isinstance(desc, str)
            assert len(desc) > 0


class TestEnsemblePrediction:
    """Test cases for EnsemblePrediction dataclass."""
    
    def test_ensemble_prediction_creation(self):
        """Test EnsemblePrediction creation."""
        pred = EnsemblePrediction(
            temperature=15.0,
            weather_class=1,
            weather_type="cloudy",
            confidence=0.85,
            baseline_contribution=0.4,
            transformer_contribution=0.6,
            method="weighted"
        )
        
        assert pred.temperature == 15.0
        assert pred.weather_type == "cloudy"
        assert pred.confidence == 0.85
        assert pred.baseline_contribution + pred.transformer_contribution == 1.0


class TestModelComparison:
    """Test cases for ModelComparison dataclass."""
    
    def test_model_comparison_creation(self):
        """Test ModelComparison creation."""
        comparison = ModelComparison(
            baseline_metrics={"temp_MAE": 2.5, "weather_Accuracy": 0.7},
            transformer_metrics={"temp_MAE": 2.0, "weather_Accuracy": 0.8},
            ensemble_metrics={"temp_MAE": 2.1, "weather_Accuracy": 0.78},
            winner_regression="Transformer",
            winner_classification="Transformer",
            summary="Transformer wins on both tasks."
        )
        
        assert comparison.winner_regression == "Transformer"
        assert comparison.baseline_metrics["temp_MAE"] > comparison.transformer_metrics["temp_MAE"]
