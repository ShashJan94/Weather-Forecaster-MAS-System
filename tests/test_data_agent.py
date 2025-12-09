"""
Tests for the Data Agent.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_agent import DataAgent, DataSplit, WeatherDataset


class TestDataAgent:
    """Test cases for DataAgent class."""
    
    def test_initialization(self):
        """Test DataAgent initialization with default parameters."""
        agent = DataAgent()
        
        assert agent.sequence_length == 7
        assert agent.cold_threshold == 5.0
        assert agent.is_fitted == False
        assert len(agent.FEATURE_COLUMNS) == 4
        assert len(agent.WEATHER_CLASSES) == 4
    
    def test_initialization_custom_params(self):
        """Test DataAgent initialization with custom parameters."""
        agent = DataAgent(sequence_length=14, cold_threshold=10.0)
        
        assert agent.sequence_length == 14
        assert agent.cold_threshold == 10.0
    
    def test_load_data_from_dataframe(self, sample_weather_df):
        """Test loading data from a DataFrame."""
        agent = DataAgent()
        df = agent.load_data(data=sample_weather_df)
        
        assert len(df) == len(sample_weather_df)
        assert agent._raw_data is not None
    
    def test_clean_data(self, sample_weather_df):
        """Test data cleaning functionality."""
        agent = DataAgent()
        
        # Add some missing values
        df = sample_weather_df.copy()
        df.loc[5, "temp"] = np.nan
        df.loc[10, "humidity"] = np.nan
        
        agent.load_data(data=df)
        cleaned = agent.clean_data()
        
        assert cleaned["temp"].isna().sum() == 0
        assert cleaned["humidity"].isna().sum() == 0
    
    def test_create_weather_labels(self, sample_weather_df):
        """Test weather label creation."""
        agent = DataAgent()
        df = agent.load_data(data=sample_weather_df)
        df = agent.clean_data(df)
        labeled_df = agent.create_weather_labels(df)
        
        assert "weather_type" in labeled_df.columns
        assert "is_cold_day" in labeled_df.columns
        assert "weather_class" in labeled_df.columns
        assert set(labeled_df["weather_type"].unique()).issubset(set(agent.WEATHER_CLASSES))
    
    def test_create_sequences(self, sample_weather_df):
        """Test sequence creation."""
        agent = DataAgent(sequence_length=7)
        df = agent.load_data(data=sample_weather_df)
        df = agent.clean_data(df)
        df = agent.create_weather_labels(df)
        
        sequences, temp_targets, class_targets, cold_targets, dates = agent.create_sequences(df)
        
        expected_samples = len(df) - agent.sequence_length
        
        assert sequences.shape == (expected_samples, 7, 4)
        assert temp_targets.shape == (expected_samples, 1)
        assert class_targets.shape == (expected_samples,)
        assert cold_targets.shape == (expected_samples, 1)
        assert len(dates) == expected_samples
    
    def test_normalize_sequences(self, sample_weather_df):
        """Test sequence normalization."""
        agent = DataAgent(sequence_length=7)
        df = agent.load_data(data=sample_weather_df)
        df = agent.clean_data(df)
        df = agent.create_weather_labels(df)
        sequences, _, _, _, _ = agent.create_sequences(df)
        
        # Fit and transform
        normalized = agent.normalize_sequences(sequences, fit=True)
        
        assert agent.is_fitted == True
        assert normalized.shape == sequences.shape
        
        # Check normalization (approximately zero mean, unit variance)
        flat = normalized.reshape(-1, 4)
        assert np.abs(flat.mean(axis=0)).max() < 0.5  # Close to zero
        assert np.abs(flat.std(axis=0) - 1.0).max() < 0.5  # Close to 1
    
    def test_split_data(self, sample_weather_df):
        """Test data splitting."""
        agent = DataAgent(sequence_length=7)
        df = agent.load_data(data=sample_weather_df)
        df = agent.clean_data(df)
        df = agent.create_weather_labels(df)
        sequences, temp_targets, class_targets, cold_targets, dates = agent.create_sequences(df)
        
        splits = agent.split_data(
            sequences, temp_targets, class_targets, cold_targets, dates,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15
        )
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        
        total_samples = len(sequences)
        assert len(splits["train"]) + len(splits["val"]) + len(splits["test"]) == total_samples
    
    def test_prepare_data_full_pipeline(self, sample_weather_df):
        """Test complete data preparation pipeline."""
        agent = DataAgent(sequence_length=7)
        
        splits, loaders = agent.prepare_data(data=sample_weather_df, batch_size=16)
        
        assert agent.is_fitted == True
        assert "train" in splits and "val" in splits and "test" in splits
        assert "train" in loaders and "val" in loaders and "test" in loaders
        
        # Check data loader
        batch = next(iter(loaders["train"]))
        assert "sequence" in batch
        assert "temp" in batch
        assert "weather_class" in batch
        assert batch["sequence"].shape[2] == 4  # num_features
    
    def test_prepare_single_sequence(self, sample_weather_df):
        """Test preparing single sequence for inference."""
        agent = DataAgent(sequence_length=7)
        agent.prepare_data(data=sample_weather_df)  # Fit scaler
        
        recent_data = sample_weather_df.tail(10)
        sequence = agent.prepare_single_sequence(recent_data)
        
        assert sequence.shape == (1, 7, 4)
        assert isinstance(sequence, torch.Tensor)
    
    def test_get_aggregated_features(self, sample_weather_df):
        """Test aggregated feature extraction for baseline model."""
        agent = DataAgent(sequence_length=7)
        splits, _ = agent.prepare_data(data=sample_weather_df)
        
        aggregated = agent.get_aggregated_features(splits)
        
        assert "train" in aggregated
        X_train, y_temp, y_class = aggregated["train"]
        
        # Should have: mean(4) + std(4) + last(4) + trend(1) = 13 features
        assert X_train.shape[1] == 13
        assert len(y_temp) == len(X_train)
        assert len(y_class) == len(X_train)
    
    def test_class_distribution(self, sample_weather_df):
        """Test class distribution calculation."""
        agent = DataAgent(sequence_length=7)
        splits, _ = agent.prepare_data(data=sample_weather_df)
        
        dist = agent.get_class_distribution(splits)
        
        assert "train" in dist
        assert all(cls in dist["train"] or dist["train"].get(cls, 0) == 0 
                   for cls in agent.WEATHER_CLASSES)


class TestWeatherDataset:
    """Test cases for WeatherDataset class."""
    
    def test_dataset_creation(self, sample_sequences):
        """Test WeatherDataset creation and indexing."""
        split = DataSplit(
            sequences=sample_sequences["sequences"],
            temp_targets=sample_sequences["temp_targets"],
            class_targets=sample_sequences["class_targets"],
            cold_targets=sample_sequences["cold_targets"],
            dates=np.arange(16)
        )
        
        dataset = WeatherDataset(split)
        
        assert len(dataset) == 16
        
        item = dataset[0]
        assert "sequence" in item
        assert "temp" in item
        assert "weather_class" in item
        assert "cold_day" in item
