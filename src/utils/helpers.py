"""
Helper functions and utilities for the Weather Forecaster system.
"""

import os
import random
import numpy as np
import torch
import joblib
from pathlib import Path
from typing import Optional, Dict, Any, Union


def set_seed(seed: int = 42) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """
    Get the best available device for PyTorch.
    Prioritizes: CUDA > MPS (Apple Silicon) > CPU
    
    Returns:
        torch.device: Best available device
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def save_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    optimizer: Optional[torch.optim.Optimizer] = None,
    epoch: Optional[int] = None,
    metrics: Optional[Dict[str, float]] = None,
    scaler: Optional[Any] = None
) -> None:
    """
    Save model checkpoint with optional training state.
    
    Args:
        model: PyTorch model to save
        path: Path to save the checkpoint
        optimizer: Optional optimizer state to save
        epoch: Optional current epoch number
        metrics: Optional dictionary of metrics
        scaler: Optional feature scaler to save
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.config if hasattr(model, "config") else None,
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    torch.save(checkpoint, path)
    
    # Save scaler separately if provided
    if scaler is not None:
        scaler_path = path.parent / f"{path.stem}_scaler.joblib"
        joblib.dump(scaler, scaler_path)


def load_model(
    model: torch.nn.Module,
    path: Union[str, Path],
    device: Optional[torch.device] = None,
    load_optimizer: bool = False,
    optimizer: Optional[torch.optim.Optimizer] = None
) -> Dict[str, Any]:
    """
    Load model checkpoint.
    
    Args:
        model: PyTorch model to load weights into
        path: Path to the checkpoint
        device: Device to load the model to
        load_optimizer: Whether to load optimizer state
        optimizer: Optimizer to load state into
        
    Returns:
        Dictionary containing loaded metadata (epoch, metrics, etc.)
    """
    path = Path(path)
    if device is None:
        device = get_device()
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    
    result = {
        "epoch": checkpoint.get("epoch"),
        "metrics": checkpoint.get("metrics"),
        "model_config": checkpoint.get("model_config")
    }
    
    if load_optimizer and optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        result["optimizer_loaded"] = True
    
    # Try to load scaler
    scaler_path = path.parent / f"{path.stem}_scaler.joblib"
    if scaler_path.exists():
        result["scaler"] = joblib.load(scaler_path)
    
    return result


def weather_code_to_label(weather_code: int) -> str:
    """
    Convert WMO weather code to weather label.
    
    WMO Weather Codes (simplified):
    0: Clear sky
    1-3: Partly cloudy
    45-48: Fog
    51-57: Drizzle
    61-67: Rain
    71-77: Snow
    80-82: Rain showers
    85-86: Snow showers
    95-99: Thunderstorm
    
    Args:
        weather_code: WMO weather interpretation code
        
    Returns:
        Weather label: sunny, cloudy, rainy, or snowy
    """
    if weather_code == 0:
        return "sunny"
    elif weather_code in [1, 2, 3, 45, 48]:
        return "cloudy"
    elif weather_code in [51, 53, 55, 56, 57, 61, 63, 65, 66, 67, 80, 81, 82, 95, 96, 99]:
        return "rainy"
    elif weather_code in [71, 73, 75, 77, 85, 86]:
        return "snowy"
    else:
        return "cloudy"  # Default fallback


def create_weather_label(
    temp: float,
    humidity: float,
    precipitation: float = 0.0,
    weather_code: Optional[int] = None,
    cold_threshold: float = 5.0
) -> Dict[str, Any]:
    """
    Create weather labels for classification.
    
    Args:
        temp: Temperature in Celsius
        humidity: Humidity percentage
        precipitation: Precipitation amount in mm
        weather_code: Optional WMO weather code
        cold_threshold: Temperature below which is considered cold
        
    Returns:
        Dictionary with weather_type and is_cold_day
    """
    # Determine weather type
    if weather_code is not None:
        weather_type = weather_code_to_label(weather_code)
    else:
        # Fallback logic based on other features
        if precipitation > 5:
            weather_type = "rainy" if temp > 0 else "snowy"
        elif precipitation > 0:
            weather_type = "rainy" if temp > 2 else "snowy"
        elif humidity > 80:
            weather_type = "cloudy"
        else:
            weather_type = "sunny" if humidity < 60 else "cloudy"
    
    # Determine if it's a cold day
    is_cold_day = temp < cold_threshold
    
    return {
        "weather_type": weather_type,
        "is_cold_day": is_cold_day
    }


def format_metrics(metrics: Dict[str, float], prefix: str = "") -> str:
    """
    Format metrics dictionary as a readable string.
    
    Args:
        metrics: Dictionary of metric names and values
        prefix: Optional prefix for each line
        
    Returns:
        Formatted string
    """
    lines = []
    for name, value in metrics.items():
        if isinstance(value, float):
            lines.append(f"{prefix}{name}: {value:.4f}")
        else:
            lines.append(f"{prefix}{name}: {value}")
    return "\n".join(lines)


def calculate_class_weights(labels: np.ndarray) -> torch.Tensor:
    """
    Calculate class weights for imbalanced classification.
    
    Args:
        labels: Array of class labels
        
    Returns:
        Tensor of class weights
    """
    classes, counts = np.unique(labels, return_counts=True)
    total = len(labels)
    weights = total / (len(classes) * counts)
    return torch.FloatTensor(weights)
