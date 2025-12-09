"""
Data Agent
Responsible for loading, cleaning, preprocessing, and preparing weather data.

This agent handles:
- Loading data from CSV files or Data Retriever
- Cleaning missing values
- Creating time windows (sequences)
- Feature normalization
- Train/validation/test splitting
- Creating PyTorch tensors
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
import joblib
import logging

from ..utils.helpers import weather_code_to_label, create_weather_label
from ..utils.config import Config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class DataSplit:
    """Container for a data split (train/val/test)."""
    sequences: torch.Tensor      # Shape: (N, seq_len, num_features)
    temp_targets: torch.Tensor   # Shape: (N, 1)
    class_targets: torch.Tensor  # Shape: (N,)
    cold_targets: torch.Tensor   # Shape: (N, 1)
    dates: np.ndarray           # Target dates
    
    def __len__(self):
        return len(self.sequences)


class WeatherDataset(Dataset):
    """PyTorch Dataset for weather data."""
    
    def __init__(self, data_split: DataSplit):
        self.sequences = data_split.sequences
        self.temp_targets = data_split.temp_targets
        self.class_targets = data_split.class_targets
        self.cold_targets = data_split.cold_targets
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return {
            "sequence": self.sequences[idx],
            "temp": self.temp_targets[idx],
            "weather_class": self.class_targets[idx],
            "cold_day": self.cold_targets[idx]
        }


class DataAgent:
    """
    Agent responsible for all data operations.
    
    This agent manages the entire data pipeline from raw data
    to ready-to-use PyTorch tensors.
    """
    
    FEATURE_COLUMNS = ["temp", "humidity", "pressure", "wind_speed"]
    WEATHER_CLASSES = ["sunny", "cloudy", "rainy", "snowy"]
    
    def __init__(
        self,
        config: Optional[Config] = None,
        sequence_length: int = 7,
        cold_threshold: float = 5.0
    ):
        """
        Initialize the Data Agent.
        
        Args:
            config: Configuration object
            sequence_length: Number of past days to use as input
            cold_threshold: Temperature below which is considered cold
        """
        self.config = config or Config()
        self.sequence_length = sequence_length
        self.cold_threshold = cold_threshold
        
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(self.WEATHER_CLASSES)
        
        self.is_fitted = False
        self._raw_data = None
        self._processed_data = None
    
    def load_data(
        self,
        filepath: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None
    ) -> pd.DataFrame:
        """
        Load weather data from file or DataFrame.
        
        Args:
            filepath: Path to CSV file
            data: Pre-loaded DataFrame
            
        Returns:
            Loaded DataFrame
        """
        if data is not None:
            self._raw_data = data.copy()
        elif filepath is not None:
            filepath = Path(filepath)
            logger.info(f"Loading data from {filepath}")
            self._raw_data = pd.read_csv(filepath)
            if "date" in self._raw_data.columns:
                self._raw_data["date"] = pd.to_datetime(self._raw_data["date"])
        else:
            raise ValueError("Either filepath or data must be provided")
        
        logger.info(f"Loaded {len(self._raw_data)} records")
        return self._raw_data
    
    def clean_data(self, df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Clean the data by handling missing values.
        
        Args:
            df: DataFrame to clean (uses loaded data if None)
            
        Returns:
            Cleaned DataFrame
        """
        if df is None:
            df = self._raw_data.copy()
        
        logger.info("Cleaning data...")
        original_len = len(df)
        
        # Ensure required columns exist
        for col in self.FEATURE_COLUMNS:
            if col not in df.columns:
                logger.warning(f"Missing column {col}, will be imputed")
        
        # Forward fill then backward fill for missing values
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].ffill().bfill()
        
        # Drop any remaining rows with NaN
        df = df.dropna(subset=self.FEATURE_COLUMNS)
        
        logger.info(f"Cleaned data: {original_len} -> {len(df)} records")
        return df
    
    def create_weather_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather type labels from weather codes or conditions.
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with weather_type and is_cold_day columns
        """
        df = df.copy()
        
        # Create weather type labels
        if "weather_code" in df.columns:
            df["weather_type"] = df["weather_code"].apply(weather_code_to_label)
        elif "weather_type" not in df.columns:
            # Infer from other features
            df["weather_type"] = df.apply(
                lambda row: create_weather_label(
                    temp=row.get("temp", 15),
                    humidity=row.get("humidity", 50),
                    precipitation=row.get("precipitation", 0)
                )["weather_type"],
                axis=1
            )
        
        # Create cold day flag
        df["is_cold_day"] = (df["temp"] < self.cold_threshold).astype(int)
        
        # Encode weather types to integers
        df["weather_class"] = self.label_encoder.transform(df["weather_type"])
        
        return df
    
    def create_sequences(
        self,
        df: pd.DataFrame,
        feature_columns: Optional[List[str]] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Create sequences for time-series prediction.
        
        For each sequence of `sequence_length` days, predict the next day's:
        - Temperature (regression)
        - Weather type (classification)
        - Cold day flag (binary classification)
        
        Args:
            df: Cleaned and labeled DataFrame
            feature_columns: Features to include in sequences
            
        Returns:
            Tuple of (sequences, temp_targets, class_targets, cold_targets, target_dates)
        """
        if feature_columns is None:
            feature_columns = self.FEATURE_COLUMNS
        
        logger.info(f"Creating sequences with length {self.sequence_length}")
        
        sequences = []
        temp_targets = []
        class_targets = []
        cold_targets = []
        target_dates = []
        
        for i in range(len(df) - self.sequence_length):
            # Input sequence: days i to i+seq_len-1
            seq = df.iloc[i:i + self.sequence_length][feature_columns].values
            
            # Target: day i+seq_len
            target_row = df.iloc[i + self.sequence_length]
            
            sequences.append(seq)
            temp_targets.append(target_row["temp"])
            class_targets.append(target_row["weather_class"])
            cold_targets.append(target_row["is_cold_day"])
            
            if "date" in df.columns:
                target_dates.append(target_row["date"])
            else:
                target_dates.append(i + self.sequence_length)
        
        sequences = np.array(sequences, dtype=np.float32)
        temp_targets = np.array(temp_targets, dtype=np.float32).reshape(-1, 1)
        class_targets = np.array(class_targets, dtype=np.int64)
        cold_targets = np.array(cold_targets, dtype=np.float32).reshape(-1, 1)
        target_dates = np.array(target_dates)
        
        logger.info(f"Created {len(sequences)} sequences of shape {sequences.shape}")
        return sequences, temp_targets, class_targets, cold_targets, target_dates
    
    def normalize_sequences(
        self,
        sequences: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize sequence features using StandardScaler.
        
        Args:
            sequences: Array of shape (N, seq_len, num_features)
            fit: Whether to fit the scaler (True for training data)
            
        Returns:
            Normalized sequences
        """
        n_samples, seq_len, n_features = sequences.shape
        
        # Reshape to 2D for scaling
        flat_sequences = sequences.reshape(-1, n_features)
        
        if fit:
            normalized = self.scaler.fit_transform(flat_sequences)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler not fitted. Call with fit=True first.")
            normalized = self.scaler.transform(flat_sequences)
        
        # Reshape back to 3D
        return normalized.reshape(n_samples, seq_len, n_features)
    
    def normalize_temperature(
        self,
        temp: np.ndarray,
        fit: bool = True
    ) -> np.ndarray:
        """
        Normalize temperature targets.
        Uses the temperature column statistics from feature scaler.
        """
        if not self.is_fitted:
            raise ValueError("Feature scaler not fitted yet")
        
        # Temperature is the first feature
        temp_mean = self.scaler.mean_[0]
        temp_std = self.scaler.scale_[0]
        
        return (temp - temp_mean) / temp_std
    
    def denormalize_temperature(self, temp: np.ndarray) -> np.ndarray:
        """Denormalize temperature predictions."""
        if not self.is_fitted:
            raise ValueError("Feature scaler not fitted yet")
        
        temp_mean = self.scaler.mean_[0]
        temp_std = self.scaler.scale_[0]
        
        return temp * temp_std + temp_mean
    
    def split_data(
        self,
        sequences: np.ndarray,
        temp_targets: np.ndarray,
        class_targets: np.ndarray,
        cold_targets: np.ndarray,
        dates: np.ndarray,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        shuffle: bool = False
    ) -> Dict[str, DataSplit]:
        """
        Split data into train/validation/test sets.
        
        For time-series, we typically don't shuffle to maintain temporal order.
        
        Args:
            sequences: Input sequences
            temp_targets: Temperature targets
            class_targets: Weather class targets
            cold_targets: Cold day targets
            dates: Target dates
            train_ratio: Fraction for training
            val_ratio: Fraction for validation
            test_ratio: Fraction for testing
            shuffle: Whether to shuffle data
            
        Returns:
            Dictionary with 'train', 'val', 'test' DataSplit objects
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        
        n_samples = len(sequences)
        
        if shuffle:
            indices = np.random.permutation(n_samples)
            sequences = sequences[indices]
            temp_targets = temp_targets[indices]
            class_targets = class_targets[indices]
            cold_targets = cold_targets[indices]
            dates = dates[indices]
        
        # Calculate split indices
        train_end = int(n_samples * train_ratio)
        val_end = int(n_samples * (train_ratio + val_ratio))
        
        splits = {}
        
        for name, start, end in [
            ("train", 0, train_end),
            ("val", train_end, val_end),
            ("test", val_end, n_samples)
        ]:
            splits[name] = DataSplit(
                sequences=torch.FloatTensor(sequences[start:end]),
                temp_targets=torch.FloatTensor(temp_targets[start:end]),
                class_targets=torch.LongTensor(class_targets[start:end]),
                cold_targets=torch.FloatTensor(cold_targets[start:end]),
                dates=dates[start:end]
            )
            logger.info(f"{name.capitalize()} set: {len(splits[name])} samples")
        
        return splits
    
    def create_data_loaders(
        self,
        splits: Dict[str, DataSplit],
        batch_size: int = 32,
        num_workers: int = 0
    ) -> Dict[str, DataLoader]:
        """
        Create PyTorch DataLoaders from data splits.
        
        Args:
            splits: Dictionary of DataSplit objects
            batch_size: Batch size for training
            num_workers: Number of worker processes
            
        Returns:
            Dictionary of DataLoaders
        """
        loaders = {}
        
        for name, split in splits.items():
            shuffle = (name == "train")
            loaders[name] = DataLoader(
                WeatherDataset(split),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=torch.cuda.is_available()
            )
        
        return loaders
    
    def prepare_data(
        self,
        filepath: Optional[Union[str, Path]] = None,
        data: Optional[pd.DataFrame] = None,
        batch_size: int = 32
    ) -> Tuple[Dict[str, DataSplit], Dict[str, DataLoader]]:
        """
        Full data preparation pipeline.
        
        Convenience method that runs all data preparation steps.
        
        Args:
            filepath: Path to data file
            data: Pre-loaded DataFrame
            batch_size: Batch size for DataLoaders
            
        Returns:
            Tuple of (splits dict, loaders dict)
        """
        # Load data
        df = self.load_data(filepath=filepath, data=data)
        
        # Clean data
        df = self.clean_data(df)
        
        # Create labels
        df = self.create_weather_labels(df)
        
        # Store processed data
        self._processed_data = df
        
        # Create sequences
        sequences, temp_targets, class_targets, cold_targets, dates = self.create_sequences(df)
        
        # Normalize (fit on all data, then split)
        sequences = self.normalize_sequences(sequences, fit=True)
        
        # Normalize temperature targets too (optional, but helps training)
        # temp_targets = self.normalize_temperature(temp_targets, fit=True)
        
        # Split data
        splits = self.split_data(
            sequences, temp_targets, class_targets, cold_targets, dates
        )
        
        # Create data loaders
        loaders = self.create_data_loaders(splits, batch_size=batch_size)
        
        return splits, loaders
    
    def prepare_single_sequence(
        self,
        df: pd.DataFrame
    ) -> torch.Tensor:
        """
        Prepare a single sequence for inference.
        
        Args:
            df: DataFrame with at least sequence_length rows
            
        Returns:
            Tensor of shape (1, seq_len, num_features)
        """
        if len(df) < self.sequence_length:
            raise ValueError(f"Need at least {self.sequence_length} rows")
        
        # Take last sequence_length rows
        seq_df = df.tail(self.sequence_length)
        
        # Extract features
        features = []
        for col in self.FEATURE_COLUMNS:
            if col in seq_df.columns:
                features.append(seq_df[col].values)
            else:
                # Use zeros for missing features
                features.append(np.zeros(self.sequence_length))
        
        sequence = np.column_stack(features).astype(np.float32)
        
        # Normalize if scaler is fitted
        if self.is_fitted:
            sequence = self.scaler.transform(sequence)
        
        # Add batch dimension
        return torch.FloatTensor(sequence).unsqueeze(0)
    
    def get_aggregated_features(
        self,
        splits: Dict[str, DataSplit]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create aggregated features for baseline models.
        
        For each sequence, compute:
        - Mean of each feature
        - Std of each feature
        - Last value of each feature
        - Trend (slope) of temperature
        
        Args:
            splits: Data splits dictionary
            
        Returns:
            Dictionary with aggregated features for each split
        """
        aggregated = {}
        
        for name, split in splits.items():
            sequences = split.sequences.numpy()
            n_samples, seq_len, n_features = sequences.shape
            
            # Compute aggregations
            mean_features = sequences.mean(axis=1)  # (N, n_features)
            std_features = sequences.std(axis=1)    # (N, n_features)
            last_features = sequences[:, -1, :]     # (N, n_features)
            first_features = sequences[:, 0, :]     # (N, n_features)
            
            # Temperature trend (difference between last and first)
            temp_trend = (last_features[:, 0] - first_features[:, 0]).reshape(-1, 1)
            
            # Combine all features
            X = np.hstack([
                mean_features,
                std_features,
                last_features,
                temp_trend
            ])
            
            y_temp = split.temp_targets.numpy().ravel()
            y_class = split.class_targets.numpy()
            
            aggregated[name] = (X, y_temp, y_class)
            logger.info(f"{name} aggregated features shape: {X.shape}")
        
        return aggregated
    
    def save_scaler(self, path: Union[str, Path]) -> None:
        """Save the fitted scaler."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "scaler": self.scaler,
            "label_encoder": self.label_encoder,
            "is_fitted": self.is_fitted,
            "sequence_length": self.sequence_length,
            "feature_columns": self.FEATURE_COLUMNS,
            "weather_classes": self.WEATHER_CLASSES
        }, path)
        logger.info(f"Saved scaler to {path}")
    
    def load_scaler(self, path: Union[str, Path]) -> None:
        """Load a fitted scaler."""
        path = Path(path)
        data = joblib.load(path)
        self.scaler = data["scaler"]
        self.label_encoder = data["label_encoder"]
        self.is_fitted = data["is_fitted"]
        self.sequence_length = data["sequence_length"]
        logger.info(f"Loaded scaler from {path}")
    
    def get_class_distribution(self, splits: Dict[str, DataSplit]) -> Dict[str, Dict[str, int]]:
        """Get class distribution for each split."""
        distributions = {}
        for name, split in splits.items():
            classes, counts = np.unique(split.class_targets.numpy(), return_counts=True)
            distributions[name] = {
                self.WEATHER_CLASSES[c]: int(cnt) 
                for c, cnt in zip(classes, counts)
            }
        return distributions
    
    @property
    def num_features(self) -> int:
        """Number of input features."""
        return len(self.FEATURE_COLUMNS)
    
    @property
    def num_classes(self) -> int:
        """Number of weather classes."""
        return len(self.WEATHER_CLASSES)


if __name__ == "__main__":
    # Test the data agent
    from pathlib import Path
    
    # Create synthetic data for testing
    from .data_retriever import generate_synthetic_data
    
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    synthetic_path = data_dir / "synthetic_weather.csv"
    
    if not synthetic_path.exists():
        generate_synthetic_data(save_path=synthetic_path)
    
    # Test data agent
    agent = DataAgent(sequence_length=7)
    splits, loaders = agent.prepare_data(filepath=synthetic_path)
    
    print("\nData splits:")
    for name, split in splits.items():
        print(f"  {name}: {len(split)} samples")
    
    print("\nClass distribution:")
    dist = agent.get_class_distribution(splits)
    for name, classes in dist.items():
        print(f"  {name}: {classes}")
    
    print("\nAggregated features for baseline:")
    aggregated = agent.get_aggregated_features(splits)
    for name, (X, y_temp, y_class) in aggregated.items():
        print(f"  {name}: X shape = {X.shape}")
