"""Feature engineering module for AQI prediction."""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Engineers features from raw AQI and weather data."""
    
    def __init__(self):
        pass
    
    def extract_time_features(self, df: pd.DataFrame, timestamp_col: str = 'timestamp') -> pd.DataFrame:
        """
        Extract time-based features from timestamp.
        
        Args:
            df: Input DataFrame
            timestamp_col: Name of timestamp column
        
        Returns:
            DataFrame with time features added
        """
        df = df.copy()
        
        if timestamp_col not in df.columns:
            logger.warning(f"Timestamp column '{timestamp_col}' not found. Creating from index.")
            df[timestamp_col] = pd.to_datetime(df.index) if isinstance(df.index, pd.DatetimeIndex) else pd.Timestamp.now()
        
        df[timestamp_col] = pd.to_datetime(df[timestamp_col])
        
        # Extract time features
        df['hour'] = df[timestamp_col].dt.hour
        df['day_of_week'] = df[timestamp_col].dt.dayofweek
        df['day_of_month'] = df[timestamp_col].dt.day
        df['month'] = df[timestamp_col].dt.month
        df['quarter'] = df[timestamp_col].dt.quarter
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Cyclical encoding for periodic features
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        return df
    
    def compute_derived_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute derived features from raw data.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with derived features added
        """
        df = df.copy()
        
        # AQI change rate and lags (if historical data available)
        if 'aqi' in df.columns:
            # Lag features (1h, 3h, 6h, 12h, 24h)
            for lag in [1, 3, 6, 12, 24]:
                df[f'aqi_lag_{lag}h'] = df['aqi'].shift(lag)
            
            # Change features
            df['aqi_lag1'] = df['aqi'].shift(1)
            df['aqi_change'] = df['aqi'] - df['aqi_lag1']
            df['aqi_change_rate'] = df['aqi_change'] / (df['aqi_lag1'] + 1e-6)  # Avoid division by zero
            
            # Rolling statistics (multiple windows)
            for window in [6, 12, 24, 48]:
                df[f'aqi_rolling_mean_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).mean()
                df[f'aqi_rolling_std_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).std()
                df[f'aqi_rolling_max_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).max()
                df[f'aqi_rolling_min_{window}h'] = df['aqi'].rolling(window=window, min_periods=1).min()
            
            # Exponential moving averages
            for span in [6, 12, 24]:
                df[f'aqi_ema_{span}h'] = df['aqi'].ewm(span=span, adjust=False).mean()
        
        # Pollutant ratios
        if 'pm25' in df.columns and 'pm10' in df.columns:
            df['pm25_pm10_ratio'] = df['pm25'] / (df['pm10'] + 1e-6)
        
        # Weather-derived features
        if 'temperature' in df.columns and 'humidity' in df.columns:
            df['heat_index'] = df['temperature'] * df['humidity'] / 100
            df['comfort_index'] = df['temperature'] - (df['humidity'] / 10)
            # Interaction features
            df['temp_humidity_interaction'] = df['temperature'] * df['humidity']
            df['temp_humidity_ratio'] = df['temperature'] / (df['humidity'] + 1e-6)
        
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
            df['wind_speed_cubed'] = df['wind_speed'] ** 3
            # Wind speed interactions
            if 'temperature' in df.columns:
                df['wind_temp_interaction'] = df['wind_speed'] * df['temperature']
            if 'humidity' in df.columns:
                df['wind_humidity_interaction'] = df['wind_speed'] * df['humidity']
        
        # Wind direction features (if available)
        if 'wind_direction' in df.columns:
            # Convert to radians for cyclical encoding
            df['wind_direction_sin'] = np.sin(np.radians(df['wind_direction']))
            df['wind_direction_cos'] = np.cos(np.radians(df['wind_direction']))
            # Wind direction categories (N, NE, E, SE, S, SW, W, NW)
            # Handle cyclical nature: 337.5-360° and 0-22.5° both map to North (0)
            def categorize_wind_direction(angle):
                """Categorize wind direction into 8 compass directions."""
                # Normalize to 0-360 range
                angle = angle % 360
                if angle < 22.5 or angle >= 337.5:
                    return 0  # N
                elif angle < 67.5:
                    return 1  # NE
                elif angle < 112.5:
                    return 2  # E
                elif angle < 157.5:
                    return 3  # SE
                elif angle < 202.5:
                    return 4  # S
                elif angle < 247.5:
                    return 5  # SW
                elif angle < 292.5:
                    return 6  # W
                else:  # 292.5 <= angle < 337.5
                    return 7  # NW
            
            df['wind_direction_category'] = df['wind_direction'].apply(categorize_wind_direction).astype(float)
        
        # Pressure features (if available)
        if 'pressure' in df.columns:
            # Pressure change rate
            df['pressure_change'] = df['pressure'].diff()
            df['pressure_change_rate'] = df['pressure_change'] / (df['pressure'].shift(1) + 1e-6)
            # Rolling pressure statistics
            for window in [6, 12, 24]:
                df[f'pressure_rolling_mean_{window}h'] = df['pressure'].rolling(window=window, min_periods=1).mean()
                df[f'pressure_rolling_std_{window}h'] = df['pressure'].rolling(window=window, min_periods=1).std()
        
        # Temperature features
        if 'temperature' in df.columns:
            # Temperature change rate
            df['temperature_change'] = df['temperature'].diff()
            df['temperature_change_rate'] = df['temperature_change'] / (df['temperature'].shift(1) + 1e-6)
            # Rolling temperature statistics
            for window in [6, 12, 24]:
                df[f'temperature_rolling_mean_{window}h'] = df['temperature'].rolling(window=window, min_periods=1).mean()
                df[f'temperature_rolling_std_{window}h'] = df['temperature'].rolling(window=window, min_periods=1).std()
        
        # Humidity features
        if 'humidity' in df.columns:
            # Humidity change rate
            df['humidity_change'] = df['humidity'].diff()
            df['humidity_change_rate'] = df['humidity_change'] / (df['humidity'].shift(1) + 1e-6)
            # Rolling humidity statistics
            for window in [6, 12, 24]:
                df[f'humidity_rolling_mean_{window}h'] = df['humidity'].rolling(window=window, min_periods=1).mean()
                df[f'humidity_rolling_std_{window}h'] = df['humidity'].rolling(window=window, min_periods=1).std()
        
        # Wind speed features
        if 'wind_speed' in df.columns:
            # Wind speed change rate
            df['wind_speed_change'] = df['wind_speed'].diff()
            df['wind_speed_change_rate'] = df['wind_speed_change'] / (df['wind_speed'].shift(1) + 1e-6)
            # Rolling wind speed statistics
            for window in [6, 12, 24]:
                df[f'wind_speed_rolling_mean_{window}h'] = df['wind_speed'].rolling(window=window, min_periods=1).mean()
                df[f'wind_speed_rolling_std_{window}h'] = df['wind_speed'].rolling(window=window, min_periods=1).std()
        
        # Multi-pollutant interactions
        if 'pm25' in df.columns and 'no2' in df.columns:
            df['pm25_no2_interaction'] = df['pm25'] * df['no2']
        if 'pm10' in df.columns and 'o3' in df.columns:
            df['pm10_o3_interaction'] = df['pm10'] * df['o3']
        if 'pm25' in df.columns and 'o3' in df.columns:
            df['pm25_o3_interaction'] = df['pm25'] * df['o3']
        
        # Air quality index categories (for target encoding)
        if 'aqi' in df.columns:
            df['aqi_category'] = pd.cut(
                df['aqi'],
                bins=[0, 50, 100, 150, 200, 300, float('inf')],
                labels=['Good', 'Moderate', 'Unhealthy_Sensitive', 'Unhealthy', 'Very_Unhealthy', 'Hazardous']
            )
        
        return df
    
    def create_target(self, df: pd.DataFrame, forecast_days: int = 3) -> pd.DataFrame:
        """
        Create target variables for future AQI predictions.
        
        Args:
            df: Input DataFrame with AQI data
            forecast_days: Number of days ahead to predict
        
        Returns:
            DataFrame with target columns added
        """
        df = df.copy()
        
        if 'aqi' not in df.columns:
            logger.warning("AQI column not found. Cannot create targets.")
            return df
        
        # Create targets by shifting (in production, future data would be available)
        # Placeholder implementation - production would use historical data with future values
        for i in range(1, forecast_days + 1):
            df[f'aqi_target_day_{i}'] = df['aqi'].shift(-i * 24)  # Assuming hourly data
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, create_targets: bool = True) -> pd.DataFrame:
        """
        Complete feature engineering pipeline.
        
        Args:
            df: Raw input DataFrame
            create_targets: Whether to create target variables
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting feature engineering pipeline...")
        
        # Extract time features
        df = self.extract_time_features(df)
        
        # Compute derived features
        df = self.compute_derived_features(df)
        
        # Create targets if requested
        if create_targets:
            df = self.create_target(df)
        
        logger.info(f"Feature engineering complete. Shape: {df.shape}")
        
        return df


if __name__ == "__main__":
    # Example usage
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'aqi': np.random.randint(0, 300, 100),
        'pm25': np.random.randint(0, 150, 100),
        'pm10': np.random.randint(0, 200, 100),
        'temperature': np.random.uniform(10, 30, 100),
        'humidity': np.random.uniform(30, 90, 100),
        'wind_speed': np.random.uniform(0, 20, 100)
    })
    
    engineer = FeatureEngineer()
    features = engineer.engineer_features(sample_data)
    print(features.head())
    print(f"\nFeature columns: {features.columns.tolist()}")
