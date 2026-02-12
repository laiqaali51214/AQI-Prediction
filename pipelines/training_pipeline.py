"""Training pipeline - fetches features, trains models, and stores in Model Registry."""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Optional
import logging
from pathlib import Path
import sys
import pickle
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import train_test_split, cross_val_score, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.feature_selection import SelectKBest, f_regression, RFE
import xgboost as xgb
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    logger.warning("LightGBM not available. Install with: pip install lightgbm")
from pipelines.mongodb_store import MongoDBStore
from config.settings import config

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Trains and evaluates ML models for AQI prediction."""
    
    def __init__(self):
        self.models = {}
        self.model_config = config['models']
        self.random_state = self.model_config['training']['random_state']
        self.test_size = self.model_config['training']['test_size']
        self.mongodb_store = MongoDBStore()
        self.scaler = None  # Will be set during data preparation
    
    def load_features(self, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Load features from MongoDB.
        
        Args:
            start_date: Optional start date filter
            end_date: Optional end date filter
        
        Returns:
            DataFrame with features
        """
        try:
            logger.info("Loading features from MongoDB...")
            features = self.mongodb_store.get_features(start_date=start_date, end_date=end_date)
            
            if features.empty:
                logger.warning("No features found in MongoDB. Trying local files...")
                return self._load_from_local()
            
            logger.info(f"Loaded {len(features)} rows from MongoDB")
            return features
            
        except Exception as e:
            logger.error(f"Error loading from MongoDB: {str(e)}")
            logger.info("Falling back to local files...")
            return self._load_from_local()
    
    def _load_from_local(self) -> pd.DataFrame:
        """Load features from local CSV files as fallback."""
        data_dir = project_root / "data" / "features"
        if not data_dir.exists():
            raise FileNotFoundError(f"Features directory not found: {data_dir}")
        
        # Try CSV files first (our current format)
        feature_files = list(data_dir.glob("features.csv"))
        if not feature_files:
            # Fallback to any CSV file
            feature_files = list(data_dir.glob("*.csv"))
        
        if not feature_files:
            # Last resort: try parquet files
            feature_files = list(data_dir.glob("*.parquet"))
        
        if not feature_files:
            raise FileNotFoundError(f"No feature files found in {data_dir}")
        
        logger.info(f"Loading features from {len(feature_files)} local file(s)...")
        dfs = []
        for file in feature_files:
            try:
                if file.suffix == '.csv':
                    df = pd.read_csv(file)
                    logger.info(f"Loaded {len(df)} rows from {file.name}")
                elif file.suffix == '.parquet':
                    df = pd.read_parquet(file)
                    logger.info(f"Loaded {len(df)} rows from {file.name}")
                else:
                    continue
                dfs.append(df)
            except Exception as e:
                logger.warning(f"Error loading {file.name}: {str(e)}")
                continue
        
        features = pd.concat(dfs, ignore_index=True)
        features = features.drop_duplicates(subset=['timestamp'], keep='last')
        features = features.sort_values('timestamp')
        
        logger.info(f"Loaded {len(features)} rows from local files")
        return features
    
    def prepare_data(self, features: pd.DataFrame) -> tuple:
        """
        Prepare data for training.
        
        Args:
            features: DataFrame with features
        
        Returns:
            Tuple of (X_train, X_test, y_train, y_test, feature_names)
        """
        # Select target (for now, predict next day AQI)
        # In real scenario, we'd have target columns from historical data
        if 'aqi' in features.columns:
            # Create target: AQI 24 hours ahead (next day)
            features = features.copy()
            features['target_aqi'] = features['aqi'].shift(-24)
            features = features.dropna(subset=['target_aqi'])
        
        # Select feature columns (exclude metadata and targets)
        exclude_cols = [
            'timestamp', 'city', 'pipeline_run_date', 'source',
            'aqi', 'target_aqi', 'aqi_category',
            'aqi_target_day_1', 'aqi_target_day_2', 'aqi_target_day_3',
            'inserted_at', '_id',  # MongoDB metadata
            # Exclude lag_24h features to prevent data leakage (target is AQI shifted 24h forward)
            'aqi_lag_24h', 'aqi_lag_24'
        ]
        
        feature_cols = [col for col in features.columns if col not in exclude_cols]
        
        # Remove datetime/timestamp columns (keep only numeric columns)
        feature_cols = [col for col in feature_cols 
                       if not pd.api.types.is_datetime64_any_dtype(features[col])]
        
        # Exclude any remaining lag_24h features (data leakage prevention)
        feature_cols = [col for col in feature_cols if 'lag_24' not in col.lower()]
        
        # Remove columns with too many missing values
        missing_threshold = 0.5
        feature_cols = [col for col in feature_cols 
                       if features[col].notna().sum() / len(features) > missing_threshold]
        
        # Ensure all columns are numeric
        X = features[feature_cols].copy()
        for col in X.columns:
            if not pd.api.types.is_numeric_dtype(X[col]):
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except:
                    X = X.drop(columns=[col])
                    feature_cols.remove(col)
        
        # Better imputation: use median for numeric columns
        # Handle columns that are completely missing or non-numeric
        from sklearn.impute import SimpleImputer
        
        # Drop columns with all missing values before imputation (SimpleImputer skips these)
        cols_with_data = X.columns[X.notna().any()].tolist()
        cols_all_missing = [col for col in X.columns if col not in cols_with_data]
        
        if cols_all_missing:
            logger.warning(f"Dropping columns with all missing values: {cols_all_missing}")
            X = X[cols_with_data]  # Keep only columns with data
            feature_cols = [col for col in feature_cols if col in cols_with_data]
        
        # Only impute if we have columns with data
        if len(X.columns) > 0:
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(X)
            # Get the actual columns that were imputed (in case imputer dropped any)
            imputed_cols = X.columns.tolist()
            X = pd.DataFrame(X_imputed, columns=imputed_cols, index=X.index)
        else:
            raise ValueError("No valid features remaining after removing missing columns")
        
        # Feature scaling for better model performance
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        self.scaler = scaler  # Save for later use in prediction
        
        y = features['target_aqi']
        
        # Feature selection to remove noise and improve performance
        # Use SelectKBest to select top features based on F-statistic
        if len(feature_cols) > 20:  # Only apply if we have many features
            logger.info(f"Applying feature selection (current features: {len(feature_cols)})...")
            try:
                # Select top 80% of features based on correlation with target
                k = max(20, int(len(feature_cols) * 0.8))
                selector = SelectKBest(score_func=f_regression, k=k)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                X = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
                feature_cols = selected_features
                self.feature_selector = selector  # Save selector for prediction
                original_count = len(X.columns) if hasattr(X, 'columns') else len(feature_cols)
                logger.info(f"Selected {len(feature_cols)} features out of {original_count} original features")
            except Exception as e:
                logger.warning(f"Feature selection failed: {str(e)}. Using all features.")
        
        # Validate target variable
        if y.nunique() <= 1:
            raise ValueError(f"Target variable has only {y.nunique()} unique value(s). Cannot train model. "
                           f"Check data quality - AQI values may be constant.")
        
        if y.std() < 0.01:
            logger.warning(f"Target variable has very low variance (std={y.std():.4f}). "
                         f"Model may overfit or predict constant values.")
        
        # Split data - use temporal split for time series data
        # Sort by timestamp if available to ensure temporal order
        if 'timestamp' in features.columns:
            features_sorted = features.sort_values('timestamp')
            split_idx = int(len(features_sorted) * (1 - self.test_size))
            train_indices = features_sorted.index[:split_idx]
            test_indices = features_sorted.index[split_idx:]
            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y.loc[train_indices]
            y_test = y.loc[test_indices]
        else:
            # Fallback to random split if no timestamp
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=self.test_size, random_state=self.random_state
            )
        
        logger.info(f"Training set: {len(X_train)} samples, {len(feature_cols)} features")
        logger.info(f"Test set: {len(X_test)} samples")
        logger.info(f"Target statistics - Mean: {y_train.mean():.2f}, Std: {y_train.std():.2f}, "
                   f"Unique values: {y_train.nunique()}")
        
        return X_train, X_test, y_train, y_test, feature_cols
    
    def train_random_forest(self, X_train, y_train) -> RandomForestRegressor:
        """Train Random Forest model with improved hyperparameters to reduce overfitting."""
        logger.info("Training Random Forest model...")
        model = RandomForestRegressor(
            n_estimators=150,  # Reduced to prevent overfitting
            max_depth=10,  # Reduced depth to prevent overfitting
            min_samples_split=20,  # Increased for stronger regularization
            min_samples_leaf=8,  # Increased for stronger regularization
            max_features='sqrt',  # Feature subset for diversity
            max_samples=0.7,  # Reduced bootstrap sample size for regularization
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        model.fit(X_train, y_train)
        return model
    
    def train_ridge(self, X_train, y_train) -> Ridge:
        """Train Ridge Regression model with cross-validation for optimal alpha."""
        logger.info("Training Ridge Regression model...")
        from sklearn.model_selection import GridSearchCV
        param_grid = {'alpha': [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]}
        model = GridSearchCV(
            Ridge(random_state=self.random_state),
            param_grid,
            cv=5,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info(f"Best Ridge alpha: {model.best_params_['alpha']}")
        return model.best_estimator_
    
    def train_xgboost(self, X_train, y_train) -> xgb.XGBRegressor:
        """Train XGBoost model with improved hyperparameters and early stopping."""
        logger.info("Training XGBoost model...")
        
        # Use validation set for early stopping
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        model = xgb.XGBRegressor(
            n_estimators=500,  # Increased with early stopping
            max_depth=6,
            learning_rate=0.03,
            subsample=0.7,
            colsample_bytree=0.7,
            min_child_weight=5,
            gamma=0.2,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0
        )
        
        # Use callbacks for early stopping (newer XGBoost versions)
        try:
            from xgboost import callback
            callbacks = [callback.EarlyStopping(rounds=20, save_best=True)]
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
                verbose=False
            )
        except (ImportError, AttributeError, TypeError):
            # Fallback: train without early stopping
            model.fit(
                X_train_fit, y_train_fit,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        
        # Retrain on full training set with best iteration if available
        best_iteration = getattr(model, 'best_iteration', None)
        if best_iteration and best_iteration > 0 and best_iteration < 500:
            # Retrain with optimal number of iterations
            model = xgb.XGBRegressor(
                n_estimators=best_iteration,
                max_depth=6,
                learning_rate=0.03,
                subsample=0.7,
                colsample_bytree=0.7,
                min_child_weight=5,
                gamma=0.2,
                reg_alpha=0.5,
                reg_lambda=2.0,
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0
            )
            model.fit(X_train, y_train)
        else:
            # If no early stopping or best_iteration not available, use the trained model
            # Optionally retrain on full dataset for consistency
            model.fit(X_train, y_train)
        
        return model
    
    def train_lightgbm(self, X_train, y_train):
        """Train LightGBM model if available using sklearn-compatible API."""
        if not LIGHTGBM_AVAILABLE:
            logger.warning("LightGBM not available. Skipping.")
            return None
        
        logger.info("Training LightGBM model...")
        
        # Use sklearn-compatible LGBMRegressor for VotingRegressor compatibility
        model = lgb.LGBMRegressor(
            n_estimators=500,
            num_leaves=31,
            learning_rate=0.03,
            feature_fraction=0.7,
            bagging_fraction=0.7,
            bagging_freq=5,
            min_child_samples=20,
            reg_alpha=0.5,
            reg_lambda=2.0,
            random_state=self.random_state,
            verbosity=-1,
            n_jobs=-1
        )
        
        # Use validation set for early stopping
        X_train_fit, X_val, y_train_fit, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=self.random_state
        )
        
        model.fit(
            X_train_fit, y_train_fit,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=20), lgb.log_evaluation(period=0)]
        )
        
        return model
    
    def evaluate_model(self, model, X_test, y_test, X_train=None, y_train=None) -> dict:
        """
        Evaluate model performance with additional metrics.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test targets
            X_train: Optional training features for overfitting check
            y_train: Optional training targets for overfitting check
        
        Returns:
            Dictionary with evaluation metrics
        """
        y_pred = model.predict(X_test)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((y_test - y_pred) / (y_test + 1e-8))) * 100
        
        metrics = {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        # Check for overfitting if training data provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_r2 = r2_score(y_train, y_train_pred)
            test_r2 = r2
            overfitting_gap = train_r2 - test_r2
            metrics['train_r2'] = float(train_r2)
            metrics['overfitting_gap'] = float(overfitting_gap)
            
            if overfitting_gap > 0.2:
                logger.warning(f"Potential overfitting detected: Train R²={train_r2:.3f}, Test R²={test_r2:.3f}, Gap={overfitting_gap:.3f}")
        
        return metrics
    
    def train_ensemble(self, X_train, X_test, y_train, y_test, base_models: dict) -> VotingRegressor:
        """Train ensemble model using voting regressor."""
        logger.info("Training ensemble model (VotingRegressor)...")
        
        from sklearn.base import is_regressor, BaseEstimator
        
        # Create voting regressor with best performing models
        estimators = []
        for name, model in base_models.items():
            if model is not None:
                # Ensure XGBoost models are properly recognized as regressors
                # Some XGBoost versions need explicit _estimator_type attribute
                if hasattr(model, '__class__'):
                    class_name = model.__class__.__name__
                    if 'XGB' in class_name or 'XGBoost' in class_name:
                        # Set _estimator_type on both instance and class if possible
                        if not hasattr(model, '_estimator_type'):
                            model._estimator_type = 'regressor'
                        elif model._estimator_type != 'regressor':
                            model._estimator_type = 'regressor'
                        # Also try setting on the class
                        if hasattr(model.__class__, '_estimator_type'):
                            model.__class__._estimator_type = 'regressor'
                        logger.info(f"Set _estimator_type='regressor' for {name} (type: {class_name})")
                
                # Validate that the model is recognized as a regressor
                try:
                    if is_regressor(model):
                        estimators.append((name, model))
                    else:
                        logger.warning(f"Skipping {name}: not recognized as a regressor (type: {type(model).__name__})")
                except Exception as e:
                    logger.warning(f"Error validating {name}: {e}. Skipping.")
        
        if len(estimators) < 2:
            logger.warning(f"Not enough valid regressors for ensemble ({len(estimators)} found). Skipping.")
            return None
        
        # Try to create and fit the ensemble
        try:
            ensemble = VotingRegressor(estimators=estimators, weights=None)
            ensemble.fit(X_train, y_train)
            return ensemble
        except ValueError as e:
            if 'should be a regressor' in str(e):
                # If XGBoost is causing issues, try without it
                logger.warning(f"Ensemble creation failed: {e}")
                logger.info("Attempting to create ensemble without XGBoost...")
                filtered_estimators = [(name, model) for name, model in estimators 
                                      if 'XGB' not in type(model).__name__]
                if len(filtered_estimators) >= 2:
                    ensemble = VotingRegressor(estimators=filtered_estimators, weights=None)
                    ensemble.fit(X_train, y_train)
                    logger.info(f"Created ensemble with {len(filtered_estimators)} models (XGBoost excluded)")
                    return ensemble
                else:
                    logger.error("Cannot create ensemble even without XGBoost. Not enough models.")
                    return None
            else:
                raise
    
    def train_all_models(self, X_train, X_test, y_train, y_test) -> dict:
        """
        Train and evaluate all models.
        
        Returns:
            Dictionary with model names and their metrics
        """
        results = {}
        base_models = {}  # Store for ensemble
        
        algorithms = self.model_config['algorithms']
        
        if 'random_forest' in algorithms:
            model = self.train_random_forest(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)
            self.models['random_forest'] = model
            base_models['random_forest'] = model
            results['random_forest'] = metrics
            logger.info(f"Random Forest - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
            if 'overfitting_gap' in metrics:
                logger.info(f"  Train R²: {metrics['train_r2']:.3f}, Overfitting gap: {metrics['overfitting_gap']:.3f}")
        
        if 'ridge_regression' in algorithms:
            model = self.train_ridge(X_train, y_train)
            metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)
            self.models['ridge_regression'] = model
            base_models['ridge_regression'] = model
            results['ridge_regression'] = metrics
            logger.info(f"Ridge Regression - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
            if 'overfitting_gap' in metrics:
                logger.info(f"  Train R²: {metrics['train_r2']:.3f}, Overfitting gap: {metrics['overfitting_gap']:.3f}")
        
        if 'xgboost' in algorithms:
            model = self.train_xgboost(X_train, y_train)
            # Ensure XGBoost model is recognized as regressor for VotingRegressor
            if not hasattr(model, '_estimator_type'):
                model._estimator_type = 'regressor'
            elif model._estimator_type != 'regressor':
                model._estimator_type = 'regressor'
            metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)
            self.models['xgboost'] = model
            base_models['xgboost'] = model
            results['xgboost'] = metrics
            logger.info(f"XGBoost - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
            if 'overfitting_gap' in metrics:
                logger.info(f"  Train R²: {metrics['train_r2']:.3f}, Overfitting gap: {metrics['overfitting_gap']:.3f}")
            
            # Log feature importance for XGBoost
            if hasattr(model, 'feature_importances_'):
                feature_importance = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)
                logger.info(f"Top 10 most important features:")
                for idx, row in feature_importance.head(10).iterrows():
                    logger.info(f"  {row['feature']}: {row['importance']:.4f}")
        
        # Train LightGBM if available
        if LIGHTGBM_AVAILABLE and 'lightgbm' in algorithms:
            model = self.train_lightgbm(X_train, y_train)
            if model is not None:
                # LGBMRegressor uses standard sklearn predict interface
                metrics = self.evaluate_model(model, X_test, y_test, X_train, y_train)
                self.models['lightgbm'] = model
                base_models['lightgbm'] = model
                results['lightgbm'] = metrics
                logger.info(f"LightGBM - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
                if 'overfitting_gap' in metrics:
                    logger.info(f"  Train R²: {metrics['train_r2']:.3f}, Overfitting gap: {metrics['overfitting_gap']:.3f}")
        
        # Train ensemble model
        if len(base_models) >= 2:
            ensemble = self.train_ensemble(X_train, X_test, y_train, y_test, base_models)
            if ensemble is not None:
                metrics = self.evaluate_model(ensemble, X_test, y_test, X_train, y_train)
                self.models['ensemble'] = ensemble
                results['ensemble'] = metrics
                logger.info(f"Ensemble - RMSE: {metrics['rmse']:.2f}, MAE: {metrics['mae']:.2f}, R²: {metrics['r2']:.3f}, MAPE: {metrics['mape']:.2f}%")
                if 'overfitting_gap' in metrics:
                    logger.info(f"  Train R²: {metrics['train_r2']:.3f}, Overfitting gap: {metrics['overfitting_gap']:.3f}")
        
        return results
    
    def select_best_model(self, results: dict) -> str:
        """
        Select best model based on RMSE.
        
        Args:
            results: Dictionary with model results
        
        Returns:
            Name of best model
        """
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        logger.info(f"Best model: {best_model[0]} with RMSE: {best_model[1]['rmse']:.2f}")
        return best_model[0]
    
    def save_model(self, model_name: str, model, metrics: dict, feature_names: list):
        """
        Save model to MongoDB and local storage.
        
        Args:
            model_name: Name of the model
            model: Trained model object
            metrics: Evaluation metrics
            feature_names: List of feature names
        """
        metadata = {
            'model_name': model_name,
            'trained_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            # Save to MongoDB
            self.mongodb_store.save_model(model_name, model, metrics, feature_names, metadata=metadata)
            logger.info(f"Model '{model_name}' saved to MongoDB")
        except Exception as e:
            logger.warning(f"Could not save to MongoDB: {str(e)}")
        
        # Also save locally as backup
        try:
            model_dir = project_root / "models" / model_name
            model_dir.mkdir(parents=True, exist_ok=True)
            
            model_path = model_dir / "model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            
            metadata_path = model_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump({
                    'model_name': model_name,
                    'metrics': metrics,
                    'feature_names': feature_names,
                    'trained_at': datetime.now().isoformat(),
                    'version': '1.0'
                }, f, indent=2)
            
            logger.info(f"Model also saved locally to {model_dir}")
        except Exception as e:
            logger.warning(f"Could not save model locally: {str(e)}")


def main():
    """Main entry point for training pipeline."""
    logger.info("Starting training pipeline...")
    
    trainer = ModelTrainer()
    
    # Load features
    features = trainer.load_features()
    
    if features.empty:
        logger.error("No features found. Please run feature pipeline first.")
        return
    
    # Prepare data
    X_train, X_test, y_train, y_test, feature_names = trainer.prepare_data(features)
    
    # Train models
    results = trainer.train_all_models(X_train, X_test, y_train, y_test)
    
    # Select best model
    best_model_name = trainer.select_best_model(results)
    best_model = trainer.models[best_model_name]
    best_metrics = results[best_model_name]
    
    # Save best model
    trainer.save_model(best_model_name, best_model, best_metrics, feature_names)
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    main()
