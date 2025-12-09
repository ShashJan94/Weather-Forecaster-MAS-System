"""
Test fixtures and configuration for pytest.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def sample_weather_df():
    """Create sample weather DataFrame for testing."""
    np.random.seed(42)
    n_days = 100
    
    dates = pd.date_range(start="2023-01-01", periods=n_days, freq="D")
    day_of_year = dates.dayofyear.values
    
    # Seasonal temperature pattern
    temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.randn(n_days) * 3
    
    df = pd.DataFrame({
        "date": dates,
        "temp": temp.round(1),
        "temp_max": (temp + 5).round(1),
        "temp_min": (temp - 5).round(1),
        "humidity": np.clip(60 + np.random.randn(n_days) * 15, 20, 100).round(1),
        "pressure": (1013 + np.random.randn(n_days) * 10).round(1),
        "wind_speed": np.abs(10 + np.random.randn(n_days) * 5).round(1),
        "precipitation": np.maximum(0, np.random.randn(n_days) * 3).round(1),
        "weather_code": np.random.choice([0, 1, 2, 3, 61, 63, 71, 73], n_days)
    })
    
    return df


@pytest.fixture
def sample_sequences():
    """Create sample sequences for model testing."""
    np.random.seed(42)
    batch_size = 16
    seq_len = 7
    num_features = 4
    
    sequences = np.random.randn(batch_size, seq_len, num_features).astype(np.float32)
    temp_targets = np.random.randn(batch_size, 1).astype(np.float32) * 10 + 15
    class_targets = np.random.randint(0, 4, batch_size)
    cold_targets = (temp_targets.ravel() < 5).astype(np.float32).reshape(-1, 1)
    
    return {
        "sequences": torch.FloatTensor(sequences),
        "temp_targets": torch.FloatTensor(temp_targets),
        "class_targets": torch.LongTensor(class_targets),
        "cold_targets": torch.FloatTensor(cold_targets)
    }


@pytest.fixture
def model_config():
    """Create model configuration for testing."""
    return {
        "num_features": 4,
        "d_model": 32,  # Smaller for faster testing
        "num_heads": 2,
        "num_layers": 1,  # Fewer layers for faster testing
        "d_ff": 64,
        "dropout": 0.1,
        "max_seq_len": 14,
        "num_classes": 4
    }


@pytest.fixture
def temp_dir(tmp_path):
    """Create temporary directory for test outputs."""
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    return tmp_path
