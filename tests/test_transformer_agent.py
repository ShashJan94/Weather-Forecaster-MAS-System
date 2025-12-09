"""
Tests for the Transformer Agent.
"""

import pytest
import numpy as np
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.transformer_agent import TransformerAgent, TrainingHistory, TransformerResults
from src.agents.data_agent import DataAgent, DataSplit, WeatherDataset
from torch.utils.data import DataLoader


class TestTransformerAgent:
    """Test cases for TransformerAgent class."""
    
    @pytest.fixture
    def data_loaders(self, sample_sequences):
        """Create data loaders for testing."""
        # Create train split
        train_split = DataSplit(
            sequences=sample_sequences["sequences"],
            temp_targets=sample_sequences["temp_targets"],
            class_targets=sample_sequences["class_targets"],
            cold_targets=sample_sequences["cold_targets"],
            dates=np.arange(16)
        )
        
        # Create smaller val split
        val_split = DataSplit(
            sequences=sample_sequences["sequences"][:8],
            temp_targets=sample_sequences["temp_targets"][:8],
            class_targets=sample_sequences["class_targets"][:8],
            cold_targets=sample_sequences["cold_targets"][:8],
            dates=np.arange(8)
        )
        
        train_loader = DataLoader(WeatherDataset(train_split), batch_size=8)
        val_loader = DataLoader(WeatherDataset(val_split), batch_size=8)
        
        return {"train": train_loader, "val": val_loader}
    
    def test_initialization(self):
        """Test TransformerAgent initialization."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        assert agent.model is not None
        assert agent.is_trained == False
        assert agent.config.d_model == 32
    
    def test_build_training_components(self):
        """Test building optimizer, scheduler, and loss."""
        agent = TransformerAgent(num_features=4, d_model=32)
        agent.build_training_components(learning_rate=1e-3, num_epochs=10)
        
        assert agent.optimizer is not None
        assert agent.scheduler is not None
        assert agent.criterion is not None
    
    def test_train(self, data_loaders):
        """Test model training."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        history = agent.train(
            data_loaders["train"],
            data_loaders["val"],
            num_epochs=2,
            early_stopping_patience=5
        )
        
        assert agent.is_trained == True
        assert isinstance(history, TrainingHistory)
        assert len(history.train_losses) == 2
        assert len(history.val_losses) == 2
    
    def test_evaluate(self, data_loaders):
        """Test model evaluation."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        agent.train(
            data_loaders["train"],
            data_loaders["val"],
            num_epochs=2
        )
        
        results = agent.evaluate(data_loaders["val"], "val")
        
        assert isinstance(results, TransformerResults)
        assert "temp_MAE" in results.metrics
        assert "weather_Accuracy" in results.metrics
        assert len(results.temp_predictions) == len(results.temp_actuals)
    
    def test_predict(self, sample_sequences):
        """Test single sequence prediction."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        # Even without training, prediction should work
        sequence = sample_sequences["sequences"][:1]
        prediction = agent.predict(sequence)
        
        assert "temperature" in prediction
        assert "weather_class" in prediction
        assert "weather_type" in prediction
        assert "class_probabilities" in prediction
        assert "is_cold_day" in prediction
    
    def test_predict_2d_input(self, sample_sequences):
        """Test prediction with 2D input (no batch dimension)."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        # 2D input without batch dimension
        sequence = sample_sequences["sequences"][0]  # Shape: (7, 4)
        prediction = agent.predict(sequence)
        
        assert "temperature" in prediction
        assert isinstance(prediction["temperature"], float)
    
    def test_save_and_load(self, data_loaders, temp_dir):
        """Test model saving and loading."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        agent.train(
            data_loaders["train"],
            data_loaders["val"],
            num_epochs=2
        )
        
        # Save model
        model_path = temp_dir / "models" / "transformer.pt"
        agent.save(model_path)
        
        # Create new agent and load
        new_agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        new_agent.load(model_path)
        
        assert new_agent.is_trained == True
        
        # Predictions should be identical
        test_input = torch.randn(1, 7, 4)
        
        agent.model.eval()
        new_agent.model.eval()
        
        with torch.no_grad():
            orig_pred = agent.predict(test_input)
            loaded_pred = new_agent.predict(test_input)
        
        assert abs(orig_pred["temperature"] - loaded_pred["temperature"]) < 1e-5
    
    def test_classification_report(self):
        """Test classification report generation."""
        agent = TransformerAgent()
        
        y_true = np.array([0, 1, 2, 3, 0, 1])
        y_pred = np.array([0, 1, 2, 2, 0, 0])
        
        report = agent.get_classification_report(y_true, y_pred)
        
        assert isinstance(report, str)
        assert "sunny" in report or "precision" in report.lower()
    
    def test_training_history_tracking(self, data_loaders):
        """Test that training history is properly tracked."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        agent.train(
            data_loaders["train"],
            data_loaders["val"],
            num_epochs=3
        )
        
        history = agent.get_training_history()
        
        assert history is not None
        assert len(history.train_losses) == 3
        assert len(history.train_metrics) == 3
        assert history.best_epoch >= 0
        assert history.best_val_loss > 0
    
    def test_get_all_metrics(self, data_loaders):
        """Test metrics collection."""
        agent = TransformerAgent(
            num_features=4,
            d_model=32,
            num_heads=2,
            num_layers=1
        )
        
        agent.train(
            data_loaders["train"],
            data_loaders["val"],
            num_epochs=2
        )
        
        agent.evaluate(data_loaders["val"], "test")
        
        all_metrics = agent.get_all_metrics()
        
        assert "test" in all_metrics


class TestTransformerResults:
    """Test cases for TransformerResults dataclass."""
    
    def test_results_summary(self):
        """Test results summary generation."""
        results = TransformerResults(
            temp_predictions=np.array([1, 2, 3]),
            temp_actuals=np.array([1, 2, 3]),
            class_predictions=np.array([0, 1, 2]),
            class_actuals=np.array([0, 1, 2]),
            class_probabilities=np.eye(4)[:3],
            metrics={"temp_MAE": 0.5, "weather_Accuracy": 0.9}
        )
        
        summary = results.summary()
        
        assert "Transformer" in summary
        assert "temp_MAE" in summary
        assert "0.5" in summary


class TestTrainingHistory:
    """Test cases for TrainingHistory dataclass."""
    
    def test_training_history_creation(self):
        """Test TrainingHistory creation."""
        history = TrainingHistory(
            train_losses=[1.0, 0.8, 0.6],
            val_losses=[1.1, 0.9, 0.7],
            train_metrics=[{"acc": 0.5}, {"acc": 0.6}, {"acc": 0.7}],
            val_metrics=[{"acc": 0.4}, {"acc": 0.5}, {"acc": 0.6}],
            best_epoch=2,
            best_val_loss=0.7
        )
        
        assert len(history.train_losses) == 3
        assert history.best_epoch == 2
        assert history.best_val_loss == 0.7
