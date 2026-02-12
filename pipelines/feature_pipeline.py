"""Feature pipeline - fetches data, engineers features, and stores in Feature Store."""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Optional
import logging
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.data_fetcher import OpenMeteoDataFetcher
from pipelines.feature_engineering import FeatureEngineer
from pipelines.data_cleaning import DataCleaner
from pipelines.aqi_calculator import EPAAQICalculator
from pipelines.mongodb_store import MongoDBStore
from config.settings import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class FeaturePipeline:
    """Main feature pipeline orchestrator."""
    
    def __init__(self):
        self.data_fetcher = OpenMeteoDataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.data_cleaner = DataCleaner()
        self.aqi_calculator = EPAAQICalculator()
        self.mongodb_store = MongoDBStore()
    
    def run(self, date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Run the feature pipeline for a specific date.
        
        Args:
            date: Date to fetch data for (defaults to current date/time)
        
        Returns:
            DataFrame with engineered features
        """
        logger.info("Starting feature pipeline...")
        
        # Fetch raw data
        logger.info("Fetching raw data from APIs...")
        raw_data = self.data_fetcher.fetch_combined_data()
        
        if raw_data.empty:
            logger.error("No data fetched from APIs. Exiting.")
            return pd.DataFrame()
        
        # Save raw data to local system
        self._save_raw_data(raw_data)
        
        # Clean data
        logger.info("Cleaning data...")
        cleaned_data = self.data_cleaner.clean_data(raw_data)
        
        # Calculate AQI if not present (using EPA method)
        if 'aqi' not in cleaned_data.columns or cleaned_data['aqi'].isna().all():
            logger.info("Calculating AQI using EPA method...")
            cleaned_data = self.aqi_calculator.calculate_aqi_from_dataframe(cleaned_data)
            # Use calculated AQI if original is missing
            if 'aqi' not in cleaned_data.columns:
                cleaned_data['aqi'] = cleaned_data.get('aqi_calculated', np.nan)
        
        # Engineer features
        logger.info("Engineering features...")
        features = self.feature_engineer.engineer_features(cleaned_data, create_targets=False)
        
        # Add metadata
        features['pipeline_run_date'] = datetime.now()
        features['city'] = config['city']['name']
        
        logger.info(f"Feature pipeline complete. Generated {len(features)} rows with {len(features.columns)} features.")
        
        return features
    
    def store_features(self, features: pd.DataFrame):
        """
        Store features in MongoDB and maintain local CSV backup.
        
        Args:
            features: DataFrame with engineered features
        """
        # Save to CSV first (local backup)
        self._save_to_csv(features)
        
        try:
            metadata = {
                'pipeline_run_date': datetime.now().isoformat(),
                'city': config['city']['name'],
                'num_features': len(features.columns),
                'num_records': len(features)
            }
            
            self.mongodb_store.insert_features(features, metadata=metadata)
            logger.info("Features successfully stored in MongoDB.")
            
        except Exception as e:
            logger.error(f"Error storing features in MongoDB: {str(e)}")
            logger.info("Features saved to CSV as backup.")
    
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
    
    def _save_raw_data(self, raw_data: pd.DataFrame):
        """
        Save raw data to local CSV file with consistent schema.
        
        Args:
            raw_data: DataFrame with raw data from APIs
        """
        raw_dir = project_root / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        
        # Normalize schema before saving
        normalized_data = self._normalize_raw_data_schema(raw_data)
        
        # Save to single consolidated CSV file
        consolidated_file = raw_dir / "raw_data.csv"
        
        if consolidated_file.exists():
            try:
                existing_df = pd.read_csv(consolidated_file)
                # Normalize existing data to ensure schema consistency
                existing_df = self._normalize_raw_data_schema(existing_df)
                
                # Combine with new data
                combined_df = pd.concat([existing_df, normalized_data], ignore_index=True)
                
                # Use DataCleaner to remove duplicates
                combined_df = self.data_cleaner.remove_duplicates(
                    combined_df,
                    subset=['timestamp'] if 'timestamp' in combined_df.columns else None
                )
                
                # Sort by timestamp if available
                if 'timestamp' in combined_df.columns:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                    combined_df = combined_df.sort_values('timestamp')
                
                # Ensure schema consistency before saving
                combined_df = self._normalize_raw_data_schema(combined_df)
                combined_df.to_csv(consolidated_file, index=False)
                logger.info(f"Updated raw data CSV: {consolidated_file} ({len(combined_df)} total records)")
            except Exception as e:
                logger.warning(f"Error updating raw data CSV: {str(e)}. Creating new file.")
                normalized_data.to_csv(consolidated_file, index=False)
        else:
            normalized_data.to_csv(consolidated_file, index=False)
            logger.info(f"Created raw data CSV: {consolidated_file}")
    
    def _normalize_features_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Normalize DataFrame to standard features schema with consistent columns.
        This ensures all feature records have the same columns even if some are missing.
        
        Args:
            df: Input DataFrame with features
            
        Returns:
            DataFrame with standardized columns
        """
        if df.empty:
            return df
        
        # Get all possible columns from existing data and new data
        # This will be dynamic based on what features are engineered
        # We'll preserve all columns that exist in either existing or new data
        
        # For now, we'll use the union of all columns
        # In production, you might want to define a fixed schema
        return df.copy()
    
    def _save_to_csv(self, features: pd.DataFrame):
        """
        Save features to CSV file with consistent schema.
        
        Args:
            features: DataFrame with engineered features
        """
        features_dir = project_root / "data" / "features"
        features_dir.mkdir(parents=True, exist_ok=True)
        
        # Save to single consolidated CSV file
        consolidated_file = features_dir / "features.csv"
        
        if consolidated_file.exists():
            # Load existing data
            try:
                existing_df = pd.read_csv(consolidated_file)
                
                # Get union of all columns from existing and new data
                all_columns = list(set(existing_df.columns.tolist() + features.columns.tolist()))
                
                # Normalize both DataFrames to have the same columns
                for col in all_columns:
                    if col not in existing_df.columns:
                        existing_df[col] = None
                    if col not in features.columns:
                        features[col] = None
                
                # Reorder columns consistently
                existing_df = existing_df[all_columns]
                features = features[all_columns]
                
                # Combine with new data
                combined_df = pd.concat([existing_df, features], ignore_index=True)
                
                # Use DataCleaner to remove duplicates
                combined_df = self.data_cleaner.remove_duplicates(
                    combined_df, 
                    subset=['timestamp'] if 'timestamp' in combined_df.columns else None
                )
                
                # Sort by timestamp if available
                if 'timestamp' in combined_df.columns:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'], errors='coerce')
                    combined_df = combined_df.sort_values('timestamp')
                
                # Save consolidated file
                combined_df.to_csv(consolidated_file, index=False)
                logger.info(f"Updated features CSV: {consolidated_file} ({len(combined_df)} total records, {len(all_columns)} columns)")
            except Exception as e:
                logger.warning(f"Error updating features CSV: {str(e)}. Creating new file.")
                features.to_csv(consolidated_file, index=False)
        else:
            # Create new consolidated file
            features.to_csv(consolidated_file, index=False)
            logger.info(f"Created features CSV: {consolidated_file} ({len(features.columns)} columns)")


def main():
    """Main entry point for feature pipeline."""
    pipeline = FeaturePipeline()
    
    # Run pipeline
    features = pipeline.run()
    
    if not features.empty:
        # Store features
        pipeline.store_features(features)
        logger.info("Feature pipeline completed successfully!")
    else:
        logger.error("Feature pipeline failed - no features generated.")


if __name__ == "__main__":
    main()
