"""MongoDB storage module for features and models."""
import os
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional, Dict, List
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure, OperationFailure
import json
import pickle
from bson import ObjectId
from config.settings import config

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MongoDBStore:
    """MongoDB storage for features and models."""
    
    def __init__(self):
        """Initialize MongoDB connection."""
        self.mongodb_config = config.get('mongodb', {})
        self.connection_string = self.mongodb_config.get('connection_string') or os.getenv('MONGODB_URI')
        self.database_name = self.mongodb_config.get('database_name', 'aqi_predictor')
        self.features_collection_name = self.mongodb_config.get('features_collection', 'aqi_features')
        self.models_collection_name = self.mongodb_config.get('models_collection', 'aqi_models')
        self.metadata_collection_name = self.mongodb_config.get('metadata_collection', 'pipeline_metadata')
        
        self.client = None
        self.db = None
        self._connect()
    
    def _connect(self):
        """Establish MongoDB connection."""
        if not self.connection_string:
            logger.warning("MongoDB connection string not provided. Using local fallback.")
            self.connection_string = "mongodb://localhost:27017/"
        
        try:
            # MongoDB Atlas requires SSL/TLS (automatically enabled for mongodb+srv://)
            # Increase timeouts for better reliability
            connection_options = {
                'serverSelectionTimeoutMS': 30000,  # 30 seconds
                'connectTimeoutMS': 30000,  # 30 seconds
                'socketTimeoutMS': 30000,  # 30 seconds
                'retryWrites': True,
                'retryReads': True,
            }
            
            # For mongodb+srv://, TLS is automatically enabled
            # Only add TLS options for standard mongodb:// connections
            if 'mongodb://' in self.connection_string and 'mongodb+srv://' not in self.connection_string:
                connection_options['tls'] = True
            
            self.client = MongoClient(self.connection_string, **connection_options)
            self.db = self.client[self.database_name]
            logger.info(f"Connected to MongoDB database: {self.database_name}")
        except ConnectionFailure as e:
            logger.error(f"Failed to connect to MongoDB: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"MongoDB connection error: {str(e)}")
            raise
    
    def insert_features(self, features_df: pd.DataFrame, metadata: Optional[Dict] = None):
        """
        Insert features into MongoDB with duplicate checking.
        
        Args:
            features_df: DataFrame with features
            metadata: Optional metadata dictionary
        """
        if self.db is None:
            logger.error("MongoDB connection not established")
            return
        
        try:
            collection = self.db[self.features_collection_name]
            
            # Convert DataFrame to list of dictionaries
            records = features_df.to_dict('records')
            
            # Add metadata
            for record in records:
                if 'timestamp' in record:
                    if isinstance(record['timestamp'], pd.Timestamp):
                        record['timestamp'] = record['timestamp'].to_pydatetime()
                record['inserted_at'] = datetime.now()
                if metadata:
                    record['metadata'] = metadata
            
            # Check for duplicates before inserting (based on timestamp)
            if 'timestamp' in features_df.columns:
                # Get existing timestamps from MongoDB
                existing_timestamps = set()
                try:
                    existing_docs = collection.find(
                        {'timestamp': {'$in': [r['timestamp'] for r in records if 'timestamp' in r]}},
                        {'timestamp': 1, '_id': 0}
                    )
                    existing_timestamps = {doc['timestamp'] for doc in existing_docs if 'timestamp' in doc}
                except Exception as e:
                    logger.warning(f"Error checking for duplicates: {str(e)}. Proceeding with insert.")
                
                # Filter out records with duplicate timestamps
                new_records = [
                    r for r in records 
                    if 'timestamp' not in r or r['timestamp'] not in existing_timestamps
                ]
                
                if len(new_records) < len(records):
                    skipped = len(records) - len(new_records)
                    logger.info(f"Skipped {skipped} duplicate records (based on timestamp)")
                
                if new_records:
                    # Insert only new records
                    result = collection.insert_many(new_records)
                    logger.info(f"Inserted {len(result.inserted_ids)} new feature records into MongoDB")
                else:
                    logger.info("All records already exist in MongoDB. No new records inserted.")
            else:
                # No timestamp column, insert all (with risk of duplicates)
                logger.warning("No timestamp column found. Inserting all records without duplicate check.")
                result = collection.insert_many(records)
                logger.info(f"Inserted {len(result.inserted_ids)} feature records into MongoDB")
            
            # Also save to CSV for local backup
            self._save_to_csv(features_df)
            
        except Exception as e:
            logger.error(f"Error inserting features: {str(e)}")
            raise
    
    def _save_to_csv(self, features_df: pd.DataFrame):
        """
        Save features to CSV file for local backup.
        
        Args:
            features_df: DataFrame with features
        """
        try:
            from pathlib import Path
            from pipelines.data_cleaning import DataCleaner
            
            project_root = Path(__file__).parent.parent
            data_dir = project_root / "data" / "features"
            data_dir.mkdir(parents=True, exist_ok=True)
            
            # Save to consolidated CSV
            consolidated_file = data_dir / "features_consolidated.csv"
            
            if consolidated_file.exists():
                # Load existing data
                existing_df = pd.read_csv(consolidated_file)
                # Combine with new data
                combined_df = pd.concat([existing_df, features_df], ignore_index=True)
                # Use DataCleaner to remove duplicates
                cleaner = DataCleaner()
                combined_df = cleaner.remove_duplicates(
                    combined_df,
                    subset=['timestamp'] if 'timestamp' in combined_df.columns else None
                )
                # Sort by timestamp if available
                if 'timestamp' in combined_df.columns:
                    combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
                    combined_df = combined_df.sort_values('timestamp')
                # Save consolidated file
                combined_df.to_csv(consolidated_file, index=False)
            else:
                # Create new consolidated file
                features_df.to_csv(consolidated_file, index=False)
        except Exception as e:
            logger.warning(f"Error saving to CSV: {str(e)}")
    
    def _ensure_connection(self):
        """Ensure MongoDB connection is active, reconnect if needed."""
        try:
            if self.client is None or self.db is None:
                logger.info("Reconnecting to MongoDB...")
                self._connect()
            else:
                # Test connection
                self.client.admin.command('ping')
        except Exception as e:
            logger.warning(f"Connection lost, reconnecting... Error: {str(e)[:100]}")
            self._connect()
    
    def get_features(self, 
                    start_date: Optional[datetime] = None,
                    end_date: Optional[datetime] = None,
                    limit: Optional[int] = None,
                    sort_by: str = 'timestamp') -> pd.DataFrame:
        """
        Retrieve features from MongoDB.
        
        Args:
            start_date: Start date filter
            end_date: End date filter
            limit: Maximum number of records to return
            sort_by: Field to sort by
        
        Returns:
            DataFrame with features
        """
        # Ensure connection is active before querying
        self._ensure_connection()
        
        if self.db is None:
            logger.error("MongoDB connection not established")
            return pd.DataFrame()
        
        try:
            collection = self.db[self.features_collection_name]
            
            # Build query
            query = {}
            if start_date or end_date:
                query['timestamp'] = {}
                if start_date:
                    query['timestamp']['$gte'] = start_date
                if end_date:
                    query['timestamp']['$lte'] = end_date
            
            # Execute query - avoid sorting on large collections to prevent memory errors
            # If limit is specified, we can sort; otherwise fetch all and sort in pandas
            if limit and limit < 10000:
                # For small queries, sort in MongoDB
                try:
                    cursor = collection.find(query).sort(sort_by, -1).limit(limit)
                    records = list(cursor)
                except Exception as e:
                    logger.warning(f"MongoDB sort failed, fetching without sort: {str(e)}")
                    cursor = collection.find(query).limit(limit)
                    records = list(cursor)
                    # Sort in pandas if needed
                    if records:
                        df_temp = pd.DataFrame(records)
                        if sort_by in df_temp.columns:
                            df_temp = df_temp.sort_values(sort_by, ascending=False)
                            records = df_temp.to_dict('records')
            else:
                # For large queries, fetch without sort and sort in pandas
                cursor = collection.find(query)
                if limit:
                    cursor = cursor.limit(limit)
                records = list(cursor)
                # Sort in pandas
                if records and sort_by:
                    df_temp = pd.DataFrame(records)
                    if sort_by in df_temp.columns:
                        df_temp = df_temp.sort_values(sort_by, ascending=False)
                        records = df_temp.to_dict('records')
            
            if not records:
                logger.warning("No features found in MongoDB")
                return pd.DataFrame()
            
            # Remove MongoDB _id and convert to DataFrame
            for record in records:
                record.pop('_id', None)
            
            df = pd.DataFrame(records)
            
            # Convert timestamp if present
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            logger.info(f"Retrieved {len(df)} feature records from MongoDB")
            return df
            
        except Exception as e:
            logger.error(f"Error retrieving features: {str(e)[:200]}")
            return pd.DataFrame()
    
    def save_model(self, model_name: str, model, metrics: Dict, feature_names: List[str], scaler=None, feature_selector=None, metadata: Optional[Dict] = None):
        """
        Save model to MongoDB.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Evaluation metrics
            feature_names: List of feature names
            scaler: Fitted StandardScaler object (if used)
            feature_selector: Fitted feature selector object (if used)
            metadata: Optional metadata
        """
        if self.db is None:
            logger.error("MongoDB connection not established")
            return
        
        try:
            collection = self.db[self.models_collection_name]
            
            # Serialize model
            model_bytes = pickle.dumps(model)
            
            # Serialize scaler if present
            scaler_bytes = pickle.dumps(scaler) if scaler else None
            
            # Serialize feature selector if present
            selector_bytes = pickle.dumps(feature_selector) if feature_selector else None
            
            # Create document
            document = {
                'model_name': model_name,
                'model_data': model_bytes,
                'scaler_data': scaler_bytes,
                'feature_selector_data': selector_bytes,
                'metrics': metrics,
                'feature_names': feature_names,
                'trained_at': datetime.now(),
                'version': '1.0',
                'metadata': metadata or {}
            }
            
            # Insert or update with increased timeout for large models
            result = collection.update_one(
                {'model_name': model_name},
                {'$set': document},
                upsert=True
            )
            
            logger.info(f"Saved model '{model_name}' to MongoDB")
            
        except Exception as e:
            error_msg = str(e)
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                logger.warning(f"Model save to MongoDB timed out (model may be too large). "
                             f"Model is still saved locally. Error: {error_msg[:200]}")
                # Don't raise - allow training to continue since local save succeeded
            else:
                logger.error(f"Error saving model: {error_msg[:200]}")
                raise
    
    def load_model(self, model_name: str):
        """
        Load model from MongoDB.
        
        Args:
            model_name: Name of the model to load
        
        Returns:
            Tuple of (model, metrics, feature_names, metadata)
        """
        if self.db is None:
            logger.error("MongoDB connection not established")
            return None, None, None, None
        
        try:
            collection = self.db[self.models_collection_name]
            
            document = collection.find_one({'model_name': model_name})
            
            if not document:
                logger.warning(f"Model '{model_name}' not found in MongoDB")
                return None, None, None, None
            
            # Deserialize model
            model = pickle.loads(document['model_data'])
            metrics = document.get('metrics', {})
            feature_names = document.get('feature_names', [])
            metadata = document.get('metadata', {})
            
            logger.info(f"Loaded model '{model_name}' from MongoDB")
            return model, metrics, feature_names, metadata
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            return None, None, None, None
    
    def get_latest_model(self):
        """Get the most recently trained model with valid metrics."""
        if self.db is None:
            logger.error("MongoDB connection not established")
            return None, None, None, None
        
        try:
            collection = self.db[self.models_collection_name]
            # Get all models sorted by trained_at (newest first)
            all_models = list(collection.find({}, {'model_name': 1, 'metrics': 1, 'trained_at': 1})
                             .sort('trained_at', -1))
            
            if not all_models:
                return None, None, None, None
            
            # Find the latest model with valid metrics
            for model_doc in all_models:
                metrics = model_doc.get('metrics', {})
                rmse = metrics.get('rmse', float('inf'))
                model_name = model_doc.get('model_name', 'unknown')
                
                # Only return models with valid RMSE
                if isinstance(rmse, (int, float)) and rmse > 0:
                    # Load the full model document
                    document = collection.find_one({'model_name': model_name})
                    if not document:
                        continue
                    
                    model = pickle.loads(document['model_data'])
                    feature_names = document.get('feature_names', [])
                    metadata = document.get('metadata', {})
                    metadata['model_name'] = document.get('model_name', 'unknown')
                    
                    # Also load scaler and feature selector if available
                    if 'scaler_data' in document and document['scaler_data']:
                        scaler = pickle.loads(document['scaler_data'])
                        metadata['scaler'] = scaler
                    if 'feature_selector_data' in document and document['feature_selector_data']:
                        feature_selector = pickle.loads(document['feature_selector_data'])
                        metadata['feature_selector'] = feature_selector
                    
                    logger.info(f"Loaded latest valid model '{metadata.get('model_name')}' with RMSE: {metrics.get('rmse', 'N/A'):.2f}")
                    return model, metrics, feature_names, metadata
                else:
                    logger.warning(f"Skipping latest model '{model_name}' with invalid RMSE: {rmse}")
            
            # If no valid models found, return None
            logger.error("No models with valid metrics found")
            return None, None, None, None
            
        except Exception as e:
            logger.error(f"Error loading latest model: {str(e)}")
            return None, None, None, None
    
    def get_best_model(self):
        """Get the model with the best (lowest) RMSE."""
        if self.db is None:
            logger.error("MongoDB connection not established")
            return None, None, None, None
        
        try:
            collection = self.db[self.models_collection_name]
            # Get all models with metrics
            all_models = list(collection.find({}, {'model_name': 1, 'metrics': 1, 'trained_at': 1}))
            
            if not all_models:
                return None, None, None, None
            
            # Find model with lowest RMSE (excluding invalid RMSE values like 0)
            best_model = None
            best_rmse = float('inf')
            
            logger.info(f"Evaluating {len(all_models)} models to find best model...")
            for model_doc in all_models:
                metrics = model_doc.get('metrics', {})
                rmse = metrics.get('rmse', float('inf'))
                model_name = model_doc.get('model_name', 'unknown')
                # Filter out invalid RMSE values (0, None, negative, or non-numeric)
                if isinstance(rmse, (int, float)) and rmse > 0:
                    logger.debug(f"Model '{model_name}': RMSE = {rmse:.2f}")
                    if rmse < best_rmse:
                        best_rmse = rmse
                        best_model = model_doc
                        logger.debug(f"New best model: '{model_name}' with RMSE: {rmse:.2f}")
                else:
                    logger.warning(f"Skipping model '{model_name}' with invalid RMSE: {rmse}")
            
            if not best_model:
                # Fallback to latest if no metrics found
                return self.get_latest_model()
            
            # Load the best model
            document = collection.find_one({'model_name': best_model['model_name']})
            if not document:
                return None, None, None, None
            
            model = pickle.loads(document['model_data'])
            metrics = document.get('metrics', {})
            feature_names = document.get('feature_names', [])
            metadata = document.get('metadata', {})
            metadata['model_name'] = document.get('model_name', 'unknown')
            
            # Also load scaler and feature selector if available
            if 'scaler_data' in document and document['scaler_data']:
                scaler = pickle.loads(document['scaler_data'])
                metadata['scaler'] = scaler
            if 'feature_selector_data' in document and document['feature_selector_data']:
                feature_selector = pickle.loads(document['feature_selector_data'])
                metadata['feature_selector'] = feature_selector
            
            logger.info(f"Loaded best model '{metadata.get('model_name')}' with RMSE: {metrics.get('rmse', 'N/A'):.2f}")
            return model, metrics, feature_names, metadata
            
        except Exception as e:
            logger.error(f"Error loading best model: {str(e)}")
            # Fallback to latest model
            return self.get_latest_model()
    
    def save_pipeline_metadata(self, pipeline_name: str, metadata: Dict):
        """
        Save pipeline execution metadata.
        
        Args:
            pipeline_name: Name of the pipeline
            metadata: Metadata dictionary
        """
        if self.db is None:
            logger.error("MongoDB connection not established")
            return
        
        try:
            collection = self.db[self.metadata_collection_name]
            
            document = {
                'pipeline_name': pipeline_name,
                'executed_at': datetime.now(),
                **metadata
            }
            
            collection.insert_one(document)
            logger.info(f"Saved metadata for pipeline '{pipeline_name}'")
            
        except Exception as e:
            logger.error(f"Error saving pipeline metadata: {str(e)}")
    
    def export_all_to_csv(self, output_file: Optional[str] = None) -> str:
        """
        Export all features from MongoDB to CSV file.
        
        Args:
            output_file: Optional output file path. Defaults to data/features/features_export.csv
        
        Returns:
            Path to exported CSV file
        """
        from pathlib import Path
        project_root = Path(__file__).parent.parent
        
        if output_file is None:
            data_dir = project_root / "data" / "features"
            data_dir.mkdir(parents=True, exist_ok=True)
            output_file = str(data_dir / "features_export.csv")
        
        try:
            features_df = self.get_features()
            
            if features_df.empty:
                logger.warning("No features found in MongoDB to export")
                return output_file
            
            features_df.to_csv(output_file, index=False)
            logger.info(f"Exported {len(features_df)} records to CSV: {output_file}")
            return output_file
        except Exception as e:
            logger.error(f"Error exporting to CSV: {str(e)}")
            raise
    
    def close(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
