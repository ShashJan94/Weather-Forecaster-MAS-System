"""
Configuration settings for the Weather Forecaster system.
Centralizes all hyperparameters and settings for easy modification.
"""

import os
from dataclasses import dataclass, field
from typing import List, Dict, Any
from pathlib import Path


@dataclass
class TransformerConfig:
    """Configuration for the Tiny Transformer model."""
    d_model: int = 64           # Model dimension
    num_heads: int = 2          # Number of attention heads
    num_layers: int = 2         # Number of transformer encoder layers
    d_ff: int = 128             # Feed-forward dimension
    dropout: float = 0.1        # Dropout rate
    max_seq_len: int = 14       # Maximum sequence length (days)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    batch_size: int = 32
    learning_rate: float = 1e-3
    num_epochs: int = 30
    early_stopping_patience: int = 5
    weight_decay: float = 1e-4
    lambda_regression: float = 1.0    # Weight for MSE loss
    lambda_classification: float = 1.0 # Weight for CrossEntropy loss
    train_split: float = 0.7
    val_split: float = 0.15
    test_split: float = 0.15


@dataclass
class DataConfig:
    """Configuration for data processing."""
    sequence_length: int = 7    # Number of past days to use
    features: List[str] = field(default_factory=lambda: [
        "temp", "humidity", "pressure", "wind_speed"
    ])
    target_temp: str = "temp"
    target_class: str = "weather_type"
    weather_classes: List[str] = field(default_factory=lambda: [
        "sunny", "cloudy", "rainy", "snowy"
    ])
    cold_threshold: float = 5.0  # Temperature below which is considered "cold"


@dataclass
class APIConfig:
    """Configuration for weather data API."""
    base_url: str = "https://api.open-meteo.com/v1/forecast"
    historical_url: str = "https://archive-api.open-meteo.com/v1/archive"
    # Default location: Warsaw, Poland
    default_latitude: float = 52.2297
    default_longitude: float = 21.0122
    timezone: str = "Europe/Warsaw"


@dataclass
class Config:
    """Main configuration class aggregating all settings."""
    
    # Sub-configurations
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    api: APIConfig = field(default_factory=APIConfig)
    
    # Paths
    project_root: Path = field(default_factory=lambda: Path(__file__).parent.parent.parent)
    
    @property
    def data_dir(self) -> Path:
        return self.project_root / "data"
    
    @property
    def raw_data_dir(self) -> Path:
        return self.data_dir / "raw"
    
    @property
    def processed_data_dir(self) -> Path:
        return self.data_dir / "processed"
    
    @property
    def models_dir(self) -> Path:
        return self.project_root / "models"
    
    @property
    def logs_dir(self) -> Path:
        return self.project_root / "logs"
    
    def create_directories(self):
        """Create all necessary directories."""
        for dir_path in [self.raw_data_dir, self.processed_data_dir, 
                         self.models_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "transformer": {
                "d_model": self.transformer.d_model,
                "num_heads": self.transformer.num_heads,
                "num_layers": self.transformer.num_layers,
                "d_ff": self.transformer.d_ff,
                "dropout": self.transformer.dropout,
                "max_seq_len": self.transformer.max_seq_len
            },
            "training": {
                "batch_size": self.training.batch_size,
                "learning_rate": self.training.learning_rate,
                "num_epochs": self.training.num_epochs,
                "early_stopping_patience": self.training.early_stopping_patience,
                "lambda_regression": self.training.lambda_regression,
                "lambda_classification": self.training.lambda_classification
            },
            "data": {
                "sequence_length": self.data.sequence_length,
                "features": self.data.features,
                "weather_classes": self.data.weather_classes,
                "cold_threshold": self.data.cold_threshold
            }
        }


# Global config instance
config = Config()
