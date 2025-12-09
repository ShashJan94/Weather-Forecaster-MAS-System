"""
Tests for the Baseline Agent (XGBoost/RandomForest).
"""

import pytest
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.baseline_agent import BaselineAgent, BaselineResults


class TestBaselineAgent:
    """Test cases for BaselineAgent class."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing."""
        np.random.seed(42)
        n_samples = 100
        n_features = 13  # Aggregated features
        
        X = np.random.randn(n_samples, n_features)
        y_temp = np.random.randn(n_samples) * 10 + 15
        y_class = np.random.randint(0, 4, n_samples)
        
        return X, y_temp, y_class
    
    def test_initialization_xgboost(self):
        """Test initialization with XGBoost."""
        agent = BaselineAgent(use_xgboost=True)
        
        assert agent.model_name == "XGBoost"
        assert agent.is_trained == False
    
    def test_initialization_random_forest(self):
        """Test initialization with Random Forest."""
        agent = BaselineAgent(use_xgboost=False)
        
        assert agent.model_name == "RandomForest"
        assert agent.is_trained == False
    
    def test_train(self, sample_data):
        """Test model training."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        # Split data
        train_size = 70
        X_train, X_val = X[:train_size], X[train_size:]
        y_temp_train, y_temp_val = y_temp[:train_size], y_temp[train_size:]
        y_class_train, y_class_val = y_class[:train_size], y_class[train_size:]
        
        results = agent.train(
            X_train, y_temp_train, y_class_train,
            X_val, y_temp_val, y_class_val
        )
        
        assert agent.is_trained == True
        assert "train" in results
        assert "val" in results
    
    def test_predict(self, sample_data):
        """Test prediction."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        
        temp_pred, class_pred, class_probs = agent.predict(X)
        
        assert len(temp_pred) == len(X)
        assert len(class_pred) == len(X)
        assert class_probs.shape == (len(X), 4)
    
    def test_predict_before_training(self, sample_data):
        """Test that prediction fails before training."""
        agent = BaselineAgent()
        X, _, _ = sample_data
        
        with pytest.raises(ValueError, match="not trained"):
            agent.predict(X)
    
    def test_evaluate(self, sample_data):
        """Test model evaluation."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        results = agent.evaluate(X, y_temp, y_class, "test")
        
        assert "regression" in results
        assert "classification" in results
        assert isinstance(results["regression"], BaselineResults)
        assert "MAE" in results["regression"].metrics
        assert "Accuracy" in results["classification"].metrics
    
    def test_feature_importance(self, sample_data):
        """Test feature importance extraction."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        importance = agent.get_feature_importance()
        
        assert "regressor" in importance
        assert "classifier" in importance
        assert len(importance["regressor"]) == X.shape[1]
    
    def test_classification_report(self, sample_data):
        """Test classification report generation."""
        agent = BaselineAgent()
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        _, class_pred, _ = agent.predict(X)
        
        report = agent.get_classification_report(y_class, class_pred)
        
        assert isinstance(report, str)
        assert "precision" in report.lower() or "accuracy" in report.lower()
    
    def test_save_and_load_models(self, sample_data, temp_dir):
        """Test model saving and loading."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        
        # Save
        models_dir = temp_dir / "models"
        agent.save_models(models_dir)
        
        # Create new agent and load
        new_agent = BaselineAgent(use_xgboost=True)
        new_agent.load_models(models_dir)
        
        assert new_agent.is_trained == True
        
        # Predictions should be identical
        orig_pred, _, _ = agent.predict(X)
        loaded_pred, _, _ = new_agent.predict(X)
        
        np.testing.assert_array_almost_equal(orig_pred, loaded_pred)
    
    def test_predict_single(self, sample_data):
        """Test single sample prediction."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        agent.train(X, y_temp, y_class)
        
        single_sample = X[0]
        result = agent.predict_single(single_sample)
        
        assert "temperature" in result
        assert "weather_class" in result
        assert "weather_type" in result
        assert "class_probabilities" in result
        assert result["weather_type"] in agent.WEATHER_CLASSES
    
    def test_get_all_metrics(self, sample_data):
        """Test metrics collection."""
        agent = BaselineAgent(use_xgboost=True)
        X, y_temp, y_class = sample_data
        
        # Train with validation
        train_size = 70
        agent.train(
            X[:train_size], y_temp[:train_size], y_class[:train_size],
            X[train_size:], y_temp[train_size:], y_class[train_size:]
        )
        
        agent.test(X[train_size:], y_temp[train_size:], y_class[train_size:])
        
        all_metrics = agent.get_all_metrics()
        
        assert "train" in all_metrics
        assert "val" in all_metrics
        assert "test" in all_metrics


class TestBaselineResults:
    """Test cases for BaselineResults dataclass."""
    
    def test_results_summary(self):
        """Test results summary generation."""
        results = BaselineResults(
            model_name="Test Model",
            task="regression",
            predictions=np.array([1, 2, 3]),
            actuals=np.array([1, 2, 3]),
            metrics={"MAE": 0.5, "RMSE": 0.7}
        )
        
        summary = results.summary()
        
        assert "Test Model" in summary
        assert "regression" in summary
        assert "MAE" in summary
        assert "0.5" in summary
