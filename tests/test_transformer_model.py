"""
Tests for the Transformer Model.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.transformer_model import (
    TinyWeatherTransformer,
    TransformerModelConfig,
    PositionalEncoding,
    WeatherLoss
)


class TestPositionalEncoding:
    """Test cases for PositionalEncoding module."""
    
    def test_output_shape(self):
        """Test that positional encoding preserves input shape."""
        d_model = 64
        max_len = 100
        batch_size = 8
        seq_len = 14
        
        pe = PositionalEncoding(d_model, max_len)
        x = torch.randn(batch_size, seq_len, d_model)
        output = pe(x)
        
        assert output.shape == x.shape
    
    def test_different_positions_have_different_encodings(self):
        """Test that different positions have different encodings."""
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=100, dropout=0.0)
        
        x = torch.zeros(1, 10, d_model)
        output = pe(x)
        
        # Check that positions have different encodings
        pos_0 = output[0, 0, :].numpy()
        pos_5 = output[0, 5, :].numpy()
        
        assert not np.allclose(pos_0, pos_5)
    
    def test_dropout_applied(self):
        """Test that dropout is applied during training."""
        d_model = 64
        pe = PositionalEncoding(d_model, max_len=100, dropout=0.5)
        pe.train()
        
        x = torch.ones(8, 14, d_model)
        
        # Run multiple times and check for variation
        outputs = [pe(x).sum().item() for _ in range(5)]
        
        # With dropout, outputs should vary
        assert len(set(outputs)) > 1


class TestTinyWeatherTransformer:
    """Test cases for TinyWeatherTransformer model."""
    
    def test_model_creation_default_config(self):
        """Test model creation with default configuration."""
        model = TinyWeatherTransformer()
        
        assert model.config.num_features == 4
        assert model.config.d_model == 64
        assert model.config.num_heads == 2
        assert model.config.num_layers == 2
    
    def test_model_creation_custom_config(self, model_config):
        """Test model creation with custom configuration."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        
        assert model.config.d_model == 32
        assert model.config.num_layers == 1
    
    def test_forward_pass_shape(self, model_config, sample_sequences):
        """Test forward pass output shapes."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        
        x = sample_sequences["sequences"]
        outputs = model(x)
        
        batch_size = x.shape[0]
        
        assert outputs["temp_pred"].shape == (batch_size, 1)
        assert outputs["weather_logits"].shape == (batch_size, 4)
        assert outputs["cold_day_logit"].shape == (batch_size, 1)
        assert outputs["features"].shape == (batch_size, model_config["d_model"])
    
    def test_predict_method(self, model_config, sample_sequences):
        """Test prediction method."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        
        x = sample_sequences["sequences"]
        outputs = model.predict(x, return_probs=True)
        
        batch_size = x.shape[0]
        
        assert "weather_probs" in outputs
        assert "cold_day_prob" in outputs
        assert "weather_pred" in outputs
        assert outputs["weather_probs"].shape == (batch_size, 4)
        
        # Check that probabilities sum to 1
        prob_sums = outputs["weather_probs"].sum(dim=1)
        assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5)
    
    def test_model_parameters_count(self, model_config):
        """Test model parameter counting."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        
        params = model.get_num_parameters()
        
        assert "total" in params
        assert "trainable" in params
        assert params["total"] == params["trainable"]  # All params trainable
        assert params["total"] > 0
    
    def test_gradient_flow(self, model_config, sample_sequences):
        """Test that gradients flow through the model."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        
        x = sample_sequences["sequences"]
        outputs = model(x)
        
        # Backward pass
        loss = outputs["temp_pred"].mean() + outputs["weather_logits"].mean()
        loss.backward()
        
        # Check gradients exist
        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
    
    def test_model_eval_mode(self, model_config, sample_sequences):
        """Test model behavior in eval mode."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        model.eval()
        
        x = sample_sequences["sequences"]
        
        with torch.no_grad():
            out1 = model(x)
            out2 = model(x)
        
        # In eval mode, outputs should be deterministic
        assert torch.allclose(out1["temp_pred"], out2["temp_pred"])
    
    def test_single_sample_inference(self, model_config):
        """Test inference with a single sample."""
        config = TransformerModelConfig(**model_config)
        model = TinyWeatherTransformer(config)
        model.eval()
        
        # Single sample
        x = torch.randn(1, 7, 4)
        
        with torch.no_grad():
            outputs = model.predict(x)
        
        assert outputs["temp_pred"].shape == (1, 1)
        assert outputs["weather_pred"].shape == (1,)


class TestWeatherLoss:
    """Test cases for WeatherLoss module."""
    
    def test_loss_computation(self, model_config, sample_sequences):
        """Test loss computation."""
        loss_fn = WeatherLoss()
        
        outputs = {
            "temp_pred": sample_sequences["temp_targets"] + torch.randn_like(sample_sequences["temp_targets"]) * 0.1,
            "weather_logits": torch.randn(16, 4),
            "cold_day_logit": torch.randn(16, 1)
        }
        
        targets = {
            "temp": sample_sequences["temp_targets"],
            "weather_class": sample_sequences["class_targets"],
            "cold_day": sample_sequences["cold_targets"]
        }
        
        total_loss, components = loss_fn(outputs, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert total_loss.ndim == 0  # Scalar
        assert "temp_loss" in components
        assert "weather_loss" in components
        assert "cold_loss" in components
        assert "total_loss" in components
    
    def test_loss_weights(self, sample_sequences):
        """Test that loss weights affect total loss."""
        outputs = {
            "temp_pred": sample_sequences["temp_targets"],
            "weather_logits": torch.randn(16, 4),
            "cold_day_logit": torch.randn(16, 1)
        }
        
        targets = {
            "temp": sample_sequences["temp_targets"],
            "weather_class": sample_sequences["class_targets"],
            "cold_day": sample_sequences["cold_targets"]
        }
        
        loss_fn1 = WeatherLoss(lambda_temp=1.0, lambda_weather=1.0)
        loss_fn2 = WeatherLoss(lambda_temp=2.0, lambda_weather=1.0)
        
        loss1, _ = loss_fn1(outputs, targets)
        loss2, _ = loss_fn2(outputs, targets)
        
        # Different weights should give different losses
        # (unless temp_loss happens to be exactly 0)
        assert loss1.item() != loss2.item() or abs(loss1.item()) < 1e-6
    
    def test_class_weights(self, sample_sequences):
        """Test loss with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0])
        loss_fn = WeatherLoss(class_weights=class_weights)
        
        outputs = {
            "temp_pred": sample_sequences["temp_targets"],
            "weather_logits": torch.randn(16, 4),
            "cold_day_logit": torch.randn(16, 1)
        }
        
        targets = {
            "temp": sample_sequences["temp_targets"],
            "weather_class": sample_sequences["class_targets"],
            "cold_day": sample_sequences["cold_targets"]
        }
        
        total_loss, _ = loss_fn(outputs, targets)
        
        assert isinstance(total_loss, torch.Tensor)
        assert not torch.isnan(total_loss)
