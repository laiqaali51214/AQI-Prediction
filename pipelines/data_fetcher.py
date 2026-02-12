"""Data fetching module for AQI and weather data using Open-Meteo API."""
import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional, List
import logging
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class OpenMeteoDataFetcher:
    """Fetches AQI and weather data from Open-Meteo APIs."""
    
    def __init__(self):
        import os
        self.openmeteo_config = config.get('apis', {}).get('openmeteo', {})
        self.api_key = self.openmeteo_config.get('api_key', '') or os.getenv('OPENMETEO_API_KEY', '')
        self.air_quality_base_url = self.openmeteo_config.get(
            'air_quality_url', 
            "https://air-quality-api.open-meteo.com/v1/air-quality"
        )
        self.weather_base_url = self.openmeteo_config.get(
            'weather_url',
            "https://api.open-meteo.com/v1/forecast"
        )
        self.historical_base_url = self.openmeteo_config.get(
            'historical_url',
            "https://archive-api.open-meteo.com/v1/archive"
        )
        self.city = config['city']
        # Ensure timezone is explicitly set to Asia/Karachi
        self.timezone = self.city.get('timezone', 'Asia/Karachi')
        logger.info(f"Initialized OpenMeteoDataFetcher for {self.city.get('name', 'Unknown')} (lat: {self.city.get('latitude')}, lon: {self.city.get('longitude')}, timezone: {self.timezone})")
    
    def _build_params(self, lat: float, lon: float, params: Dict) -> Dict:
        """Build request parameters with optional API key."""
        base_params = {
            'latitude': lat,
            'longitude': lon,
            **params
        }
        
        # Add API key if provided (for commercial use)
        if self.api_key:
            base_params['apikey'] = self.api_key
        
        return base_params
    
    def fetch_air_quality(self, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict:
        """
        Fetch hourly air quality data from Open-Meteo Air Quality API.
        
        Args:
            lat: Latitude (defaults to config city)
            lon: Longitude (defaults to config city)
        
        Returns:
            Dictionary containing air quality data (PM2.5, PM10, CO, NO2, O3, SO2)
        """
        lat = lat or self.city['latitude']
        lon = lon or self.city['longitude']
        
        try:
            params = self._build_params(lat, lon, {
                'hourly': 'pm2_5,pm10,carbon_monoxide,nitrogen_dioxide,ozone,sulphur_dioxide',
                'timezone': self.timezone
            })
            
            response = requests.get(self.air_quality_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'hourly' in data:
                hourly = data['hourly']
                time_index = 0  # Get current hour data
                
                return {
                    'pm25': hourly.get('pm2_5', [None])[time_index] if hourly.get('pm2_5') else None,
                    'pm10': hourly.get('pm10', [None])[time_index] if hourly.get('pm10') else None,
                    'co': hourly.get('carbon_monoxide', [None])[time_index] if hourly.get('carbon_monoxide') else None,
                    'no2': hourly.get('nitrogen_dioxide', [None])[time_index] if hourly.get('nitrogen_dioxide') else None,
                    'o3': hourly.get('ozone', [None])[time_index] if hourly.get('ozone') else None,
                    'so2': hourly.get('sulphur_dioxide', [None])[time_index] if hourly.get('sulphur_dioxide') else None,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'openmeteo_air_quality'
                }
            else:
                logger.error("No hourly data in Open-Meteo Air Quality API response")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo air quality data: {str(e)}")
            return {}
    
    def fetch_weather_forecast(self, lat: Optional[float] = None, lon: Optional[float] = None) -> Dict:
        """
        Fetch weather forecast data from Open-Meteo Weather API.
        
        Args:
            lat: Latitude (defaults to config city)
            lon: Longitude (defaults to config city)
        
        Returns:
            Dictionary containing weather data (temperature, humidity, wind speed/direction)
        """
        lat = lat or self.city['latitude']
        lon = lon or self.city['longitude']
        
        try:
            params = self._build_params(lat, lon, {
                'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl',
                'timezone': self.timezone
            })
            
            response = requests.get(self.weather_base_url, params=params, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            
            if 'hourly' in data:
                hourly = data['hourly']
                time_index = 0  # Get current hour data
                
                return {
                    'temperature': hourly.get('temperature_2m', [None])[time_index] if hourly.get('temperature_2m') else None,
                    'humidity': hourly.get('relative_humidity_2m', [None])[time_index] if hourly.get('relative_humidity_2m') else None,
                    'wind_speed': hourly.get('wind_speed_10m', [None])[time_index] if hourly.get('wind_speed_10m') else None,
                    'wind_direction': hourly.get('wind_direction_10m', [None])[time_index] if hourly.get('wind_direction_10m') else None,
                    'pressure': hourly.get('pressure_msl', [None])[time_index] if hourly.get('pressure_msl') else None,
                    'timestamp': datetime.now().isoformat(),
                    'source': 'openmeteo_weather'
                }
            else:
                logger.error("No hourly data in Open-Meteo Weather API response")
                return {}
                
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo weather data: {str(e)}")
            return {}
    
    def fetch_historical_weather(self, 
                                 start_date: datetime, 
                                 end_date: datetime,
                                 lat: Optional[float] = None, 
                                 lon: Optional[float] = None) -> pd.DataFrame:
        """
        Fetch historical weather data from Open-Meteo Archive API for backfilling.
        
        Args:
            start_date: Start date for historical data
            end_date: End date for historical data (will be capped to yesterday if it's today or future)
            lat: Latitude (defaults to config city)
            lon: Longitude (defaults to config city)
        
        Returns:
            DataFrame with historical weather data
        """
        lat = lat or self.city['latitude']
        lon = lon or self.city['longitude']
        
        # Ensure end_date is not today or in the future (use yesterday as maximum)
        today = datetime.now().date()
        if end_date.date() >= today:
            end_date = datetime.combine(today - timedelta(days=1), datetime.min.time())
            logger.info(f"End date adjusted to yesterday ({end_date.date()}) to exclude current day")
        
        try:
            params = self._build_params(lat, lon, {
                'start_date': start_date.strftime('%Y-%m-%d'),
                'end_date': end_date.strftime('%Y-%m-%d'),
                'hourly': 'temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,pressure_msl',
                'timezone': self.timezone
            })
            
            response = requests.get(self.historical_base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            
            if 'hourly' in data:
                hourly = data['hourly']
                times = hourly.get('time', [])
                
                # Create DataFrame
                df_data = {
                    'timestamp': pd.to_datetime(times),
                    'temperature': hourly.get('temperature_2m', []),
                    'humidity': hourly.get('relative_humidity_2m', []),
                    'wind_speed': hourly.get('wind_speed_10m', []),
                    'wind_direction': hourly.get('wind_direction_10m', []),
                    'pressure': hourly.get('pressure_msl', [])
                }
                
                df = pd.DataFrame(df_data)
                df['source'] = 'openmeteo_historical'
                
                logger.info(f"Fetched {len(df)} historical weather records from {start_date.date()} to {end_date.date()}")
                return df
            else:
                logger.error("No hourly data in Open-Meteo Historical API response")
                return pd.DataFrame()
                
        except Exception as e:
            logger.error(f"Error fetching Open-Meteo historical data: {str(e)}")
            return pd.DataFrame()
    
    def fetch_combined_data(self) -> pd.DataFrame:
        """
        Fetch and combine current air quality and weather data.
        
        Returns:
            DataFrame with combined AQI and weather data with standardized columns
        """
        aqi_data = self.fetch_air_quality()
        weather_data = self.fetch_weather_forecast()
        
        # Combine data
        combined = {**aqi_data, **weather_data}
        
        # Remove duplicate timestamp and source
        if 'timestamp' in combined:
            combined['timestamp'] = datetime.now()
        if 'source' in combined:
            combined['source'] = 'openmeteo_combined'
        
        # Create DataFrame
        df = pd.DataFrame([combined])
        
        # Normalize to standard schema
        df = self._normalize_raw_data_schema(df)
        
        return df
    
    @staticmethod
    def _normalize_raw_data_schema(df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard raw data schema with consistent columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with standardized columns
        """
        if df.empty:
            return df
        
        # Define standard schema for raw data
        standard_columns = [
            'pm25', 'pm10', 'no2', 'o3', 'co', 'so2',
            'temperature', 'humidity', 'wind_speed', 'wind_direction', 'pressure',
            'timestamp', 'source'
        ]
        
        # Create new DataFrame with standard columns
        normalized_df = pd.DataFrame(index=df.index, columns=standard_columns)
        
        # Copy existing columns that match standard schema
        for col in standard_columns:
            if col in df.columns:
                normalized_df[col] = df[col].values
            else:
                # Fill missing columns with None
                normalized_df[col] = None
        
        # Ensure timestamp is datetime
        if 'timestamp' in normalized_df.columns:
            normalized_df['timestamp'] = pd.to_datetime(normalized_df['timestamp'], errors='coerce')
        
        return normalized_df


# Alias for backward compatibility
AQIDataFetcher = OpenMeteoDataFetcher


if __name__ == "__main__":
    fetcher = OpenMeteoDataFetcher()
    data = fetcher.fetch_combined_data()
    print(data)
    
    # Test historical data fetch
    end_date = datetime.now()
    start_date = end_date - timedelta(days=7)
    historical = fetcher.fetch_historical_weather(start_date, end_date)
    print(f"\nHistorical data shape: {historical.shape}")
    print(historical.head())