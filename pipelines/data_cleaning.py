"""Data cleaning and preprocessing module."""
import pandas as pd
import numpy as np
from typing import Optional, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataCleaner:
    """Clean and preprocess AQI and weather data."""
    
    def __init__(self, outlier_method: str = 'iqr', percentile_cap: float = 0.99):
        """
        Initialize data cleaner.
        
        Args:
            outlier_method: Method for outlier detection ('iqr', 'zscore', 'percentile')
            percentile_cap: Percentile to cap outliers (0.99 = 99th percentile)
        """
        self.outlier_method = outlier_method
        self.percentile_cap = percentile_cap
    
    def handle_missing_values(self, df: pd.DataFrame, strategy: str = 'forward_fill') -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            strategy: Strategy for handling missing values
                - 'forward_fill': Forward fill
                - 'backward_fill': Backward fill
                - 'interpolate': Linear interpolation
                - 'mean': Fill with mean
                - 'median': Fill with median
                - 'drop': Drop rows with missing values
        
        Returns:
            DataFrame with missing values handled
        """
        df = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                if strategy == 'forward_fill':
                    df[col] = df[col].ffill()
                elif strategy == 'backward_fill':
                    df[col] = df[col].bfill()
                elif strategy == 'interpolate':
                    df[col] = df[col].interpolate(method='linear')
                elif strategy == 'mean':
                    df[col] = df[col].fillna(df[col].mean())
                elif strategy == 'median':
                    df[col] = df[col].fillna(df[col].median())
                elif strategy == 'drop':
                    df = df.dropna(subset=[col])
        
        logger.info(f"Handled missing values using strategy: {strategy}")
        return df
    
    def detect_outliers_iqr(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Detect outliers using Interquartile Range (IQR) method.
        
        Args:
            df: Input DataFrame
            columns: List of columns to check (None = all numeric columns)
        
        Returns:
            DataFrame with outlier flags added
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                Q1 = df[col].quantile(0.25)
                Q3 = df[col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                df[f'{col}_outlier'] = ((df[col] < lower_bound) | (df[col] > upper_bound)).astype(int)
        
        return df
    
    def cap_outliers_percentile(self, df: pd.DataFrame, columns: Optional[list] = None) -> pd.DataFrame:
        """
        Cap outliers at specified percentile.
        
        Args:
            df: Input DataFrame
            columns: List of columns to cap (None = all numeric columns)
        
        Returns:
            DataFrame with outliers capped
        """
        df = df.copy()
        
        if columns is None:
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        
        for col in columns:
            if col in df.columns:
                lower_percentile = (1 - self.percentile_cap) / 2
                upper_percentile = self.percentile_cap + (1 - self.percentile_cap) / 2
                
                lower_bound = df[col].quantile(lower_percentile)
                upper_bound = df[col].quantile(upper_percentile)
                
                df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
        
        logger.info(f"Capped outliers at {self.percentile_cap * 100}th percentile")
        return df
    
    def remove_duplicates(self, df: pd.DataFrame, subset: Optional[list] = None) -> pd.DataFrame:
        """
        Remove duplicate rows.
        
        Args:
            df: Input DataFrame
            subset: List of columns to consider for duplicates (None = all columns)
        
        Returns:
            DataFrame with duplicates removed
        """
        initial_len = len(df)
        df = df.drop_duplicates(subset=subset, keep='last')
        removed = initial_len - len(df)
        
        if removed > 0:
            logger.info(f"Removed {removed} duplicate rows")
        
        return df
    
    def validate_data_ranges(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate data ranges for pollutants and weather.
        
        Args:
            df: Input DataFrame
        
        Returns:
            DataFrame with validation flags
        """
        df = df.copy()
        
        # Reasonable ranges for pollutants (in µg/m³)
        ranges = {
            'pm25': (0, 500),
            'pm10': (0, 600),
            'o3': (0, 500),  # ppb
            'no2': (0, 2000),  # ppb
            'co': (0, 50),    # ppm
            'so2': (0, 1000),  # ppb
            'temperature': (-50, 60),  # Celsius
            'humidity': (0, 100),  # Percentage
            'pressure': (800, 1100),  # hPa
            'wind_speed': (0, 100)  # m/s
        }
        
        for col, (min_val, max_val) in ranges.items():
            if col in df.columns:
                invalid = ((df[col] < min_val) | (df[col] > max_val)) & df[col].notna()
                if invalid.sum() > 0:
                    logger.warning(f"Found {invalid.sum()} invalid values in {col} (range: {min_val}-{max_val})")
                    df.loc[invalid, col] = np.nan
        
        return df
    
    def drop_high_missing_columns(self, df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
        """
        Drop columns with more than specified percentage of missing values.
        
        Args:
            df: Input DataFrame
            threshold: Maximum proportion of missing values allowed (0.5 = 50%)
        
        Returns:
            DataFrame with high-missing columns removed
        """
        df = df.copy()
        initial_cols = len(df.columns)
        
        # Calculate missing value percentage for each column
        missing_percentage = df.isna().sum() / len(df)
        
        # Find columns with more than threshold missing values
        cols_to_drop = missing_percentage[missing_percentage > threshold].index.tolist()
        
        # Don't drop essential columns like timestamp
        essential_cols = ['timestamp', 'source', 'city', 'pipeline_run_date', 'backfill_date']
        cols_to_drop = [col for col in cols_to_drop if col not in essential_cols]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            logger.info(f"Dropped {len(cols_to_drop)} columns with >{threshold*100}% missing values: {cols_to_drop}")
        
        final_cols = len(df.columns)
        if initial_cols != final_cols:
            logger.info(f"Column count: {initial_cols} → {final_cols}")
        
        return df
    
    def clean_data(self, df: pd.DataFrame, 
                   handle_missing: bool = True,
                   remove_duplicates: bool = True,
                   validate_ranges: bool = True,
                   cap_outliers: bool = True,
                   drop_high_missing: bool = True) -> pd.DataFrame:
        """
        Complete data cleaning pipeline.
        
        Args:
            df: Input DataFrame
            handle_missing: Whether to handle missing values
            remove_duplicates: Whether to remove duplicates
            validate_ranges: Whether to validate data ranges
            cap_outliers: Whether to cap outliers
            drop_high_missing: Whether to drop columns with >50% missing values
        
        Returns:
            Cleaned DataFrame
        """
        logger.info("Starting data cleaning pipeline...")
        initial_len = len(df)
        initial_cols = len(df.columns)
        
        # Drop columns with >50% missing values first (before other processing)
        if drop_high_missing:
            df = self.drop_high_missing_columns(df, threshold=0.5)
        
        # Remove duplicates
        if remove_duplicates:
            df = self.remove_duplicates(df, subset=['timestamp'] if 'timestamp' in df.columns else None)
        
        # Validate ranges
        if validate_ranges:
            df = self.validate_data_ranges(df)
        
        # Handle missing values
        if handle_missing:
            df = self.handle_missing_values(df, strategy='interpolate')
        
        # Cap outliers
        if cap_outliers:
            df = self.cap_outliers_percentile(df)
        
        final_len = len(df)
        final_cols = len(df.columns)
        logger.info(f"Data cleaning complete. Rows: {initial_len} → {final_len}, Columns: {initial_cols} → {final_cols}")
        
        return df


if __name__ == "__main__":
    # Example usage
    import pandas as pd
    
    # Create sample data with issues
    data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'pm25': np.random.normal(25, 10, 100),
        'pm10': np.random.normal(50, 20, 100),
        'temperature': np.random.normal(20, 5, 100),
        'humidity': np.random.normal(60, 15, 100)
    })
    
    # Add some issues
    data.loc[10:15, 'pm25'] = np.nan  # Missing values
    data.loc[20, 'pm25'] = 1000  # Outlier
    data.loc[50:52] = data.loc[50:52]  # Duplicates
    
    print("Original data shape:", data.shape)
    print("Missing values:", data.isna().sum().sum())
    
    cleaner = DataCleaner()
    cleaned = cleaner.clean_data(data)
    
    print("\nCleaned data shape:", cleaned.shape)
    print("Missing values:", cleaned.isna().sum().sum())
