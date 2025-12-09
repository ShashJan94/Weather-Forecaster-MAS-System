"""
Data Retriever Agent
Fetches live weather data from Open-Meteo API.

This agent is responsible for:
- Fetching historical weather data
- Fetching current weather conditions
- Fetching weather forecasts
- Caching data locally
"""

import os
import json
import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataRetriever:
    """
    Agent for retrieving live weather data from Open-Meteo API.
    
    Open-Meteo is a free, open-source weather API that doesn't require
    an API key for basic usage.
    """
    
    # API endpoints
    FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
    HISTORICAL_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    # Weather variables to fetch
    HOURLY_VARIABLES = [
        "temperature_2m",
        "relative_humidity_2m", 
        "surface_pressure",
        "wind_speed_10m",
        "precipitation",
        "weather_code"
    ]
    
    DAILY_VARIABLES = [
        "temperature_2m_mean",
        "temperature_2m_max",
        "temperature_2m_min",
        "precipitation_sum",
        "wind_speed_10m_max",
        "weather_code"
    ]
    
    def __init__(
        self,
        latitude: float = 52.2297,  # Warsaw default
        longitude: float = 21.0122,
        timezone: str = "Europe/Warsaw",
        cache_dir: Optional[Path] = None
    ):
        """
        Initialize the Data Retriever.
        
        Args:
            latitude: Location latitude
            longitude: Location longitude  
            timezone: Timezone string
            cache_dir: Directory for caching data
        """
        self.latitude = latitude
        self.longitude = longitude
        self.timezone = timezone
        self.cache_dir = cache_dir or Path(__file__).parent.parent.parent / "data" / "raw"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "WeatherForecaster/1.0"
        })
    
    def set_location(self, latitude: float, longitude: float, timezone: str = "auto"):
        """Update the location for data retrieval."""
        self.latitude = latitude
        self.longitude = longitude
        if timezone != "auto":
            self.timezone = timezone
    
    def fetch_historical_data(
        self,
        start_date: str,
        end_date: str,
        daily: bool = True
    ) -> pd.DataFrame:
        """
        Fetch historical weather data.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            daily: If True, fetch daily data; otherwise hourly
            
        Returns:
            DataFrame with historical weather data
        """
        logger.info(f"Fetching historical data from {start_date} to {end_date}")
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": self.timezone
        }
        
        if daily:
            params["daily"] = ",".join(self.DAILY_VARIABLES)
        else:
            params["hourly"] = ",".join(self.HOURLY_VARIABLES)
        
        try:
            response = self.session.get(self.HISTORICAL_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if daily and "daily" in data:
                df = self._parse_daily_data(data["daily"])
            elif not daily and "hourly" in data:
                df = self._parse_hourly_data(data["hourly"])
            else:
                logger.warning("No data returned from API")
                return pd.DataFrame()
            
            logger.info(f"Successfully fetched {len(df)} records")
            return df
            
        except requests.RequestException as e:
            logger.error(f"Error fetching historical data: {e}")
            raise
    
    def fetch_forecast(self, days: int = 7) -> pd.DataFrame:
        """
        Fetch weather forecast for upcoming days.
        
        Args:
            days: Number of forecast days (1-16)
            
        Returns:
            DataFrame with forecast data
        """
        logger.info(f"Fetching {days}-day forecast")
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ",".join(self.DAILY_VARIABLES),
            "timezone": self.timezone,
            "forecast_days": min(days, 16)
        }
        
        try:
            response = self.session.get(self.FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "daily" in data:
                df = self._parse_daily_data(data["daily"])
                logger.info(f"Successfully fetched {len(df)} forecast records")
                return df
            else:
                logger.warning("No forecast data returned")
                return pd.DataFrame()
                
        except requests.RequestException as e:
            logger.error(f"Error fetching forecast: {e}")
            raise
    
    def fetch_current_weather(self) -> Dict[str, Any]:
        """
        Fetch current weather conditions.
        
        Returns:
            Dictionary with current weather data
        """
        logger.info("Fetching current weather")
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,relative_humidity_2m,weather_code,wind_speed_10m,surface_pressure",
            "timezone": self.timezone
        }
        
        try:
            response = self.session.get(self.FORECAST_URL, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if "current" in data:
                current = data["current"]
                return {
                    "timestamp": current.get("time"),
                    "temp": current.get("temperature_2m"),
                    "humidity": current.get("relative_humidity_2m"),
                    "pressure": current.get("surface_pressure"),
                    "wind_speed": current.get("wind_speed_10m"),
                    "weather_code": current.get("weather_code")
                }
            return {}
            
        except requests.RequestException as e:
            logger.error(f"Error fetching current weather: {e}")
            raise
    
    def fetch_and_save_training_data(
        self,
        years_back: int = 2,
        filename: str = "weather_history.csv"
    ) -> pd.DataFrame:
        """
        Fetch historical data for training and save to cache.
        
        Args:
            years_back: Number of years of historical data to fetch
            filename: Output filename
            
        Returns:
            DataFrame with historical weather data
        """
        end_date = datetime.now() - timedelta(days=5)  # API has ~5 day delay
        start_date = end_date - timedelta(days=365 * years_back)
        
        df = self.fetch_historical_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            daily=True
        )
        
        if not df.empty:
            filepath = self.cache_dir / filename
            df.to_csv(filepath, index=False)
            logger.info(f"Saved training data to {filepath}")
        
        return df
    
    def load_cached_data(self, filename: str = "weather_history.csv") -> pd.DataFrame:
        """
        Load cached weather data.
        
        Args:
            filename: Cached data filename
            
        Returns:
            DataFrame with cached data, or empty DataFrame if not found
        """
        filepath = self.cache_dir / filename
        
        if filepath.exists():
            logger.info(f"Loading cached data from {filepath}")
            df = pd.read_csv(filepath)
            df["date"] = pd.to_datetime(df["date"])
            return df
        else:
            logger.warning(f"No cached data found at {filepath}")
            return pd.DataFrame()
    
    def get_data_for_prediction(
        self,
        sequence_length: int = 7
    ) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """
        Get the most recent data needed for making predictions.
        
        This combines historical data with current conditions.
        
        Args:
            sequence_length: Number of past days needed
            
        Returns:
            Tuple of (sequence DataFrame, current conditions dict)
        """
        # Fetch recent historical data
        end_date = datetime.now() - timedelta(days=1)
        start_date = end_date - timedelta(days=sequence_length + 5)
        
        historical = self.fetch_historical_data(
            start_date=start_date.strftime("%Y-%m-%d"),
            end_date=end_date.strftime("%Y-%m-%d"),
            daily=True
        )
        
        # Get current conditions
        current = self.fetch_current_weather()
        
        # Take the last sequence_length days
        if len(historical) >= sequence_length:
            sequence = historical.tail(sequence_length).copy()
        else:
            logger.warning(f"Not enough historical data: got {len(historical)}, need {sequence_length}")
            sequence = historical.copy()
        
        return sequence, current
    
    def _parse_daily_data(self, daily_data: Dict) -> pd.DataFrame:
        """Parse daily data from API response."""
        df = pd.DataFrame({
            "date": pd.to_datetime(daily_data.get("time", [])),
            "temp": daily_data.get("temperature_2m_mean", []),
            "temp_max": daily_data.get("temperature_2m_max", []),
            "temp_min": daily_data.get("temperature_2m_min", []),
            "precipitation": daily_data.get("precipitation_sum", []),
            "wind_speed": daily_data.get("wind_speed_10m_max", []),
            "weather_code": daily_data.get("weather_code", [])
        })
        
        # Add derived features
        df["humidity"] = 60 + np.random.randn(len(df)) * 15  # Approximate if not available
        df["pressure"] = 1013 + np.random.randn(len(df)) * 10  # Approximate
        
        return df
    
    def _parse_hourly_data(self, hourly_data: Dict) -> pd.DataFrame:
        """Parse hourly data from API response."""
        df = pd.DataFrame({
            "datetime": pd.to_datetime(hourly_data.get("time", [])),
            "temp": hourly_data.get("temperature_2m", []),
            "humidity": hourly_data.get("relative_humidity_2m", []),
            "pressure": hourly_data.get("surface_pressure", []),
            "wind_speed": hourly_data.get("wind_speed_10m", []),
            "precipitation": hourly_data.get("precipitation", []),
            "weather_code": hourly_data.get("weather_code", [])
        })
        
        return df
    
    def get_location_info(self) -> Dict[str, Any]:
        """Get information about the current location."""
        return {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "timezone": self.timezone
        }
    
    @staticmethod
    def get_popular_locations() -> Dict[str, Tuple[float, float, str]]:
        """Get a dictionary of popular city locations."""
        return {
            "Warsaw": (52.2297, 21.0122, "Europe/Warsaw"),
            "London": (51.5074, -0.1278, "Europe/London"),
            "New York": (40.7128, -74.0060, "America/New_York"),
            "Tokyo": (35.6762, 139.6503, "Asia/Tokyo"),
            "Sydney": (-33.8688, 151.2093, "Australia/Sydney"),
            "Berlin": (52.5200, 13.4050, "Europe/Berlin"),
            "Paris": (48.8566, 2.3522, "Europe/Paris"),
            "Dubai": (25.2048, 55.2708, "Asia/Dubai"),
            "Singapore": (1.3521, 103.8198, "Asia/Singapore"),
            "Los Angeles": (34.0522, -118.2437, "America/Los_Angeles")
        }


def generate_synthetic_data(
    num_days: int = 730,  # 2 years
    start_date: str = "2022-01-01",
    save_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Generate synthetic weather data for testing.
    
    Creates realistic-looking weather data with seasonal patterns.
    
    Args:
        num_days: Number of days of data to generate
        start_date: Starting date
        save_path: Optional path to save the data
        
    Returns:
        DataFrame with synthetic weather data
    """
    np.random.seed(42)
    
    dates = pd.date_range(start=start_date, periods=num_days, freq="D")
    day_of_year = dates.dayofyear.values
    
    # Temperature with seasonal pattern (Northern Hemisphere)
    base_temp = 10 + 15 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    temp_noise = np.random.randn(num_days) * 5
    temp = base_temp + temp_noise
    
    # Temperature extremes
    temp_max = temp + np.abs(np.random.randn(num_days) * 3) + 5
    temp_min = temp - np.abs(np.random.randn(num_days) * 3) - 5
    
    # Humidity - higher in winter, lower in summer
    humidity = 70 - 20 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    humidity += np.random.randn(num_days) * 10
    humidity = np.clip(humidity, 20, 100)
    
    # Pressure
    pressure = 1013 + np.random.randn(num_days) * 15
    
    # Wind speed
    wind_speed = 10 + np.abs(np.random.randn(num_days) * 8)
    
    # Precipitation
    precip_prob = 0.3 + 0.2 * np.sin(2 * np.pi * (day_of_year + 60) / 365)
    has_precip = np.random.rand(num_days) < precip_prob
    precipitation = np.where(has_precip, np.random.exponential(5, num_days), 0)
    
    # Weather codes based on conditions
    weather_codes = []
    for i in range(num_days):
        if precipitation[i] > 0:
            if temp[i] < 0:
                weather_codes.append(np.random.choice([71, 73, 75]))  # Snow
            else:
                weather_codes.append(np.random.choice([61, 63, 65]))  # Rain
        elif humidity[i] > 80:
            weather_codes.append(np.random.choice([1, 2, 3]))  # Cloudy
        else:
            weather_codes.append(0)  # Sunny
    
    df = pd.DataFrame({
        "date": dates,
        "temp": temp.round(1),
        "temp_max": temp_max.round(1),
        "temp_min": temp_min.round(1),
        "humidity": humidity.round(1),
        "pressure": pressure.round(1),
        "wind_speed": wind_speed.round(1),
        "precipitation": precipitation.round(1),
        "weather_code": weather_codes
    })
    
    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"Saved synthetic data to {save_path}")
    
    return df


if __name__ == "__main__":
    # Test the data retriever
    retriever = DataRetriever()
    
    # Fetch current weather
    print("\nCurrent weather:")
    current = retriever.fetch_current_weather()
    print(current)
    
    # Fetch forecast
    print("\n7-day forecast:")
    forecast = retriever.fetch_forecast(days=7)
    print(forecast)
    
    # Generate synthetic data for testing
    print("\nGenerating synthetic data...")
    data_dir = Path(__file__).parent.parent.parent / "data" / "raw"
    synthetic = generate_synthetic_data(save_path=data_dir / "synthetic_weather.csv")
    print(f"Generated {len(synthetic)} days of synthetic data")
