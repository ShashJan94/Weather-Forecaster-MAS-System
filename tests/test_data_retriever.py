"""
Tests for the Data Retriever Agent.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from unittest.mock import Mock, patch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.agents.data_retriever import DataRetriever, generate_synthetic_data


class TestDataRetriever:
    """Test cases for DataRetriever class."""
    
    def test_initialization(self):
        """Test DataRetriever initialization."""
        retriever = DataRetriever()
        
        assert retriever.latitude == 52.2297  # Warsaw default
        assert retriever.longitude == 21.0122
        assert retriever.timezone == "Europe/Warsaw"
    
    def test_initialization_custom_location(self):
        """Test DataRetriever with custom location."""
        retriever = DataRetriever(
            latitude=51.5074,
            longitude=-0.1278,
            timezone="Europe/London"
        )
        
        assert retriever.latitude == 51.5074
        assert retriever.longitude == -0.1278
        assert retriever.timezone == "Europe/London"
    
    def test_set_location(self):
        """Test location setting."""
        retriever = DataRetriever()
        retriever.set_location(40.7128, -74.0060, "America/New_York")
        
        assert retriever.latitude == 40.7128
        assert retriever.longitude == -74.0060
        assert retriever.timezone == "America/New_York"
    
    def test_get_location_info(self):
        """Test getting location info."""
        retriever = DataRetriever()
        info = retriever.get_location_info()
        
        assert "latitude" in info
        assert "longitude" in info
        assert "timezone" in info
    
    def test_get_popular_locations(self):
        """Test popular locations dictionary."""
        locations = DataRetriever.get_popular_locations()
        
        assert "Warsaw" in locations
        assert "London" in locations
        assert "New York" in locations
        assert len(locations["Warsaw"]) == 3  # (lat, lon, tz)
    
    def test_parse_daily_data(self):
        """Test parsing daily API data."""
        retriever = DataRetriever()
        
        daily_data = {
            "time": ["2023-01-01", "2023-01-02"],
            "temperature_2m_mean": [5.0, 6.0],
            "temperature_2m_max": [8.0, 9.0],
            "temperature_2m_min": [2.0, 3.0],
            "precipitation_sum": [0.0, 2.5],
            "wind_speed_10m_max": [15.0, 20.0],
            "weather_code": [0, 61]
        }
        
        df = retriever._parse_daily_data(daily_data)
        
        assert len(df) == 2
        assert "temp" in df.columns
        assert "temp_max" in df.columns
        assert "precipitation" in df.columns
        assert df["temp"].iloc[0] == 5.0
    
    def test_parse_hourly_data(self):
        """Test parsing hourly API data."""
        retriever = DataRetriever()
        
        hourly_data = {
            "time": ["2023-01-01T00:00", "2023-01-01T01:00"],
            "temperature_2m": [5.0, 5.5],
            "relative_humidity_2m": [80, 82],
            "surface_pressure": [1013, 1012],
            "wind_speed_10m": [10, 12],
            "precipitation": [0.0, 0.0],
            "weather_code": [0, 0]
        }
        
        df = retriever._parse_hourly_data(hourly_data)
        
        assert len(df) == 2
        assert "temp" in df.columns
        assert "humidity" in df.columns
        assert "pressure" in df.columns
    
    @patch('requests.Session.get')
    def test_fetch_current_weather_mocked(self, mock_get):
        """Test current weather fetching with mocked response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "current": {
                "time": "2023-01-01T12:00",
                "temperature_2m": 10.5,
                "relative_humidity_2m": 75,
                "surface_pressure": 1015,
                "wind_speed_10m": 12.5,
                "weather_code": 3
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        retriever = DataRetriever()
        current = retriever.fetch_current_weather()
        
        assert current["temp"] == 10.5
        assert current["humidity"] == 75
        assert current["weather_code"] == 3
    
    @patch('requests.Session.get')
    def test_fetch_forecast_mocked(self, mock_get):
        """Test forecast fetching with mocked response."""
        mock_response = Mock()
        mock_response.json.return_value = {
            "daily": {
                "time": ["2023-01-01", "2023-01-02", "2023-01-03"],
                "temperature_2m_mean": [5.0, 6.0, 7.0],
                "temperature_2m_max": [8.0, 9.0, 10.0],
                "temperature_2m_min": [2.0, 3.0, 4.0],
                "precipitation_sum": [0.0, 2.5, 0.0],
                "wind_speed_10m_max": [15.0, 20.0, 18.0],
                "weather_code": [0, 61, 3]
            }
        }
        mock_response.raise_for_status = Mock()
        mock_get.return_value = mock_response
        
        retriever = DataRetriever()
        forecast = retriever.fetch_forecast(days=3)
        
        assert len(forecast) == 3
        assert "temp" in forecast.columns
    
    def test_load_cached_data_not_found(self, temp_dir):
        """Test loading cached data when file doesn't exist."""
        retriever = DataRetriever(cache_dir=temp_dir / "data")
        
        df = retriever.load_cached_data("nonexistent.csv")
        
        assert df.empty


class TestGenerateSyntheticData:
    """Test cases for synthetic data generation."""
    
    def test_generate_basic(self):
        """Test basic synthetic data generation."""
        df = generate_synthetic_data(num_days=100)
        
        assert len(df) == 100
        assert "date" in df.columns
        assert "temp" in df.columns
        assert "humidity" in df.columns
        assert "pressure" in df.columns
        assert "wind_speed" in df.columns
        assert "weather_code" in df.columns
    
    def test_generate_with_save(self, temp_dir):
        """Test synthetic data generation with saving."""
        save_path = temp_dir / "data" / "test_synthetic.csv"
        
        df = generate_synthetic_data(num_days=50, save_path=save_path)
        
        assert save_path.exists()
        
        # Load and verify
        loaded = pd.read_csv(save_path)
        assert len(loaded) == 50
    
    def test_synthetic_data_realistic_ranges(self):
        """Test that synthetic data has realistic value ranges."""
        df = generate_synthetic_data(num_days=365)
        
        # Temperature should be reasonable
        assert df["temp"].min() > -30
        assert df["temp"].max() < 45
        
        # Humidity should be 0-100
        assert df["humidity"].min() >= 0
        assert df["humidity"].max() <= 100
        
        # Wind speed should be positive
        assert df["wind_speed"].min() >= 0
        
        # Precipitation should be non-negative
        assert df["precipitation"].min() >= 0
    
    def test_synthetic_data_seasonal_pattern(self):
        """Test that synthetic data shows seasonal temperature pattern."""
        df = generate_synthetic_data(num_days=365, start_date="2023-01-01")
        
        # Summer months should be warmer than winter months
        df["date"] = pd.to_datetime(df["date"])
        df["month"] = df["date"].dt.month
        
        summer_temp = df[df["month"].isin([6, 7, 8])]["temp"].mean()
        winter_temp = df[df["month"].isin([12, 1, 2])]["temp"].mean()
        
        assert summer_temp > winter_temp
    
    def test_synthetic_data_weather_codes(self):
        """Test that weather codes are valid."""
        df = generate_synthetic_data(num_days=365)
        
        valid_codes = [0, 1, 2, 3, 61, 63, 65, 71, 73, 75]
        
        assert all(code in valid_codes for code in df["weather_code"].unique())
    
    def test_reproducibility(self):
        """Test that generation is reproducible with same seed."""
        df1 = generate_synthetic_data(num_days=50)
        df2 = generate_synthetic_data(num_days=50)
        
        # Both should have same values due to fixed seed
        pd.testing.assert_frame_equal(df1, df2)
