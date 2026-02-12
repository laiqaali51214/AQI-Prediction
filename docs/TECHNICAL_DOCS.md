# Technical Documentation

This document contains all technical details about the AQI Predictor project, including data structures, preprocessing, feature engineering, model improvements, and project architecture.

## Table of Contents

1. [Dataset Description](#dataset-description)
2. [Preprocessing and EDA Summary](#preprocessing-and-eda-summary)
3. [Model Performance Improvements](#model-performance-improvements)
4. [Project Structure](#project-structure)

---

## Dataset Description

### Executive Summary

This document describes the air quality and weather dataset collected for the AQI (Air Quality Index) prediction system for Karachi, Pakistan. The dataset contains **8,765 hourly records** spanning **365 days** (January 16, 2025 to January 16, 2026), providing comprehensive historical data for machine learning model training.

### Dataset Structure

#### 1. Raw Data (`data/raw/raw_data.csv`)

**Purpose**: Contains unprocessed air quality and weather measurements directly from the Open-Meteo API.

**Record Count**: 8,765 hourly observations

**Time Period**: January 16, 2025 to January 16, 2026 (1 full year)

**Data Columns**:

##### Air Quality Pollutants (µg/m³):
- **PM2.5**: Fine particulate matter (2.5 micrometers) - primary air quality indicator
- **PM10**: Coarse particulate matter (10 micrometers)
- **NO2**: Nitrogen dioxide
- **O3**: Ozone
- **CO**: Carbon monoxide (not available in historical data)
- **SO2**: Sulfur dioxide (not available in historical data)

##### Weather Parameters:
- **Temperature**: Air temperature in Celsius (°C)
- **Humidity**: Relative humidity percentage (%)
- **Wind Speed**: Wind speed in meters per second (m/s)
- **Wind Direction**: Wind direction in degrees (0-360°)
- **Pressure**: Atmospheric pressure in hectopascals (hPa)

##### Metadata:
- **Timestamp**: Date and time of observation (hourly intervals)
- **Source**: Data source identifier (`openmeteo_historical`)

**Data Characteristics**:
- **Frequency**: Hourly measurements (24 records per day)
- **Completeness**: ~99.9% data coverage (8,765 records for 365 days × 24 hours = 8,760 expected)
- **Geographic Coverage**: Karachi, Pakistan (Lat: 24.8608°, Lon: 67.0104°)
- **Data Quality**: Historical air quality values estimated from weather patterns using correlation models

#### 2. Engineered Features (`data/features/features.csv`)

**Purpose**: Contains processed and engineered features ready for machine learning model training.

**Record Count**: 8,765 records (one per raw data record)

**Total Features**: 69 engineered features

**Feature Categories**:

##### 1. Original Measurements (13 features)
- All raw pollutant and weather measurements from raw data

##### 2. Time-Based Features (12 features)
- **Hour**: Hour of day (0-23)
- **Day of Week**: Day of week (0-6, Monday=0)
- **Day of Month**: Day of month (1-31)
- **Month**: Month of year (1-12)
- **Quarter**: Quarter of year (1-4)
- **Is Weekend**: Binary indicator (0=weekday, 1=weekend)
- **Cyclical Encodings**: Sin/cos transformations for hour, day of week, and month to capture periodic patterns

##### 3. Lag Features (6 features)
- **AQI Lag 1h, 3h, 6h, 12h, 24h**: Previous AQI values at different time intervals
- Captures temporal dependencies and trends

##### 4. Rolling Statistics (24 features)
- **Rolling Mean**: 6h, 12h, 24h, 48h windows
- **Rolling Standard Deviation**: 6h, 12h, 24h, 48h windows
- **Rolling Maximum**: 6h, 12h, 24h, 48h windows
- **Rolling Minimum**: 6h, 12h, 24h, 48h windows
- Captures short-term and medium-term patterns

##### 5. Exponential Moving Averages (3 features)
- **EMA 6h, 12h, 24h**: Exponentially weighted moving averages
- Emphasizes recent observations more than older ones

##### 6. Change Features (2 features)
- **AQI Change**: Difference from previous hour
- **AQI Change Rate**: Percentage change from previous hour

##### 7. Derived Features (5 features)
- **PM25/PM10 Ratio**: Ratio of fine to coarse particles
- **Heat Index**: Temperature × Humidity / 100
- **Comfort Index**: Temperature - (Humidity / 10)
- **Wind Speed Squared**: Non-linear wind effect
- **AQI Category**: Categorical classification (Good, Moderate, Unhealthy_Sensitive, etc.)

##### 8. Individual Pollutant AQI Values (5 features)
- **PM25 AQI, PM10 AQI, NO2 AQI, O3 AQI, SO2 AQI**: Individual AQI calculations per pollutant
- **Dominant Pollutant**: The pollutant with highest AQI value

##### 9. Metadata (3 features)
- **City**: Location identifier (Karachi)
- **Pipeline Run Date**: When the feature was generated
- **Backfill Date**: Original timestamp of the data

### Data Collection Methodology

#### Data Sources:
1. **Open-Meteo Historical Weather Archive API**: Provides historical weather data (temperature, humidity, wind, pressure)
2. **Open-Meteo Air Quality API**: Provides current air quality data (for real-time predictions)
3. **Estimated Historical AQI**: Since historical air quality data is not available, AQI values are estimated using:
   - Weather pattern correlations (temperature, wind speed, humidity)
   - Diurnal variation models (time-of-day patterns)
   - Statistical variation to simulate realistic conditions

#### Data Processing Pipeline:
1. **Data Fetching**: Retrieve hourly weather and air quality data from APIs
2. **Data Cleaning**: Handle missing values, remove outliers, validate ranges
3. **Feature Engineering**: Create time-based, lag, rolling, and derived features
4. **AQI Calculation**: Compute Air Quality Index using EPA standards
5. **Data Storage**: Save to CSV files and MongoDB database

### Data Quality Metrics

- **Completeness**: 99.9% (8,765 / 8,760 expected records)
- **Consistency**: All records follow standardized schema with consistent column structure
- **Temporal Coverage**: Complete year with hourly granularity
- **Geographic Coverage**: Single location (Karachi) for focused analysis

### Technical Specifications

- **File Format**: CSV (Comma-Separated Values)
- **Encoding**: UTF-8
- **Delimiter**: Comma
- **Header Row**: Yes (column names in first row)
- **Timestamp Format**: ISO 8601 (YYYY-MM-DD HH:MM:SS)
- **Missing Values**: Represented as empty cells or NaN
- **File Size**: 
  - Raw Data: ~1.2 MB
  - Features: ~4.5 MB

---

## Preprocessing and EDA Summary

### Overview

This section summarizes all preprocessing and exploratory data analysis (EDA) steps implemented in the AQI Prediction project.

### 1. Data Preprocessing Steps

#### 1.1 Column-Level Preprocessing

##### **Drop High Missing Value Columns** (`drop_high_missing_columns`)
- **Purpose**: Remove columns with >50% missing values before processing
- **Implementation**: 
  - Calculates missing value percentage for each column
  - Drops columns exceeding 50% threshold
  - Protects essential columns (timestamp, source, city, pipeline_run_date, backfill_date)
- **Location**: `pipelines/data_cleaning.py`
- **When Applied**: First step in the cleaning pipeline (before other operations)

##### **Data Range Validation** (`validate_data_ranges`)
- **Purpose**: Identify and flag invalid values outside reasonable ranges
- **Validated Ranges**:
  - PM2.5: 0-500 µg/m³
  - PM10: 0-600 µg/m³
  - O3: 0-500 ppb
  - NO2: 0-2000 ppb
  - CO: 0-50 ppm
  - SO2: 0-1000 ppb
  - Temperature: -50 to 60°C
  - Humidity: 0-100%
  - Pressure: 800-1100 hPa
  - Wind Speed: 0-100 m/s
- **Action**: Invalid values are set to NaN
- **Location**: `pipelines/data_cleaning.py`

#### 1.2 Row-Level Preprocessing

##### **Duplicate Removal** (`remove_duplicates`)
- **Purpose**: Remove duplicate records
- **Method**: 
  - Identifies duplicates based on timestamp (or all columns if no timestamp)
  - Keeps the last occurrence (`keep='last'`)
- **Location**: `pipelines/data_cleaning.py`
- **When Applied**: Early in pipeline, after dropping high-missing columns

##### **Missing Value Handling** (`handle_missing_values`)
- **Purpose**: Handle remaining missing values after column filtering
- **Strategies Available**:
  1. **Forward Fill** (`forward_fill`): Propagate last valid value forward
  2. **Backward Fill** (`backward_fill`): Propagate next valid value backward
  3. **Interpolation** (`interpolate`): Linear interpolation between values (default)
  4. **Mean Imputation** (`mean`): Fill with column mean
  5. **Median Imputation** (`median`): Fill with column median
  6. **Drop** (`drop`): Remove rows with missing values
- **Current Strategy**: Linear interpolation
- **Location**: `pipelines/data_cleaning.py`

#### 1.3 Outlier Detection and Treatment

##### **Outlier Detection** (`detect_outliers_iqr`)
- **Purpose**: Identify outliers using Interquartile Range (IQR) method
- **Method**:
  - Calculates Q1 (25th percentile) and Q3 (75th percentile)
  - IQR = Q3 - Q1
  - Lower bound = Q1 - 1.5 × IQR
  - Upper bound = Q3 + 1.5 × IQR
  - Flags values outside bounds
- **Output**: Adds `{column}_outlier` binary flags
- **Location**: `pipelines/data_cleaning.py`

##### **Outlier Capping** (`cap_outliers_percentile`)
- **Purpose**: Cap extreme outliers instead of removing them
- **Method**:
  - Caps values at 99th percentile (default)
  - Uses `clip()` to bound values between lower and upper percentiles
  - Preserves data while reducing extreme values
- **Location**: `pipelines/data_cleaning.py`
- **When Applied**: After missing value handling

#### 1.4 Complete Preprocessing Pipeline

##### **Main Cleaning Function** (`clean_data`)
- **Execution Order**:
  1. Drop columns with >50% missing values
  2. Remove duplicate rows
  3. Validate data ranges
  4. Handle missing values (interpolation)
  5. Cap outliers at 99th percentile
- **Logging**: Tracks row and column count changes
- **Location**: `pipelines/data_cleaning.py`

### 2. Feature Engineering Steps

#### 2.1 Time-Based Features (`extract_time_features`)

##### **Basic Time Features**:
- `hour`: Hour of day (0-23)
- `day_of_week`: Day of week (0=Monday, 6=Sunday)
- `day_of_month`: Day of month (1-31)
- `month`: Month of year (1-12)
- `quarter`: Quarter of year (1-4)
- `is_weekend`: Binary indicator (0=weekday, 1=weekend)

##### **Cyclical Encoding** (for periodic patterns):
- `hour_sin`, `hour_cos`: Sin/cos transformation of hour (captures 24-hour cycles)
- `day_of_week_sin`, `day_of_week_cos`: Sin/cos transformation of day of week (captures weekly cycles)
- `month_sin`, `month_cos`: Sin/cos transformation of month (captures yearly cycles)

**Purpose**: Convert cyclical time features to continuous values that preserve periodicity

#### 2.2 Lag Features (`compute_derived_features`)

##### **AQI Lag Features**:
- `aqi_lag_1h`: AQI value 1 hour ago
- `aqi_lag_3h`: AQI value 3 hours ago
- `aqi_lag_6h`: AQI value 6 hours ago
- `aqi_lag_12h`: AQI value 12 hours ago
- `aqi_lag_24h`: AQI value 24 hours ago

**Purpose**: Capture temporal dependencies and recent trends

#### 2.3 Change Features

- `aqi_lag1`: Previous hour's AQI (alias for lag_1h)
- `aqi_change`: Difference from previous hour (current - previous)
- `aqi_change_rate`: Percentage change from previous hour

**Purpose**: Capture rate of change and momentum

#### 2.4 Rolling Statistics

##### **Rolling Windows**: 6h, 12h, 24h, 48h
For each window, calculates:
- `aqi_rolling_mean_{window}h`: Moving average
- `aqi_rolling_std_{window}h`: Moving standard deviation
- `aqi_rolling_max_{window}h`: Maximum in window
- `aqi_rolling_min_{window}h`: Minimum in window

**Purpose**: Capture short-term and medium-term patterns, trends, and volatility

#### 2.5 Exponential Moving Averages (EMA)

- `aqi_ema_6h`: 6-hour exponential moving average
- `aqi_ema_12h`: 12-hour exponential moving average
- `aqi_ema_24h`: 24-hour exponential moving average

**Purpose**: Emphasize recent observations more than older ones (weighted average)

#### 2.6 Derived Features

##### **Pollutant Ratios**:
- `pm25_pm10_ratio`: Ratio of fine to coarse particles

##### **Weather-Derived Features**:
- `heat_index`: Temperature × Humidity / 100
- `comfort_index`: Temperature - (Humidity / 10)
- `wind_speed_squared`: Non-linear wind effect (wind²)

##### **Categorical Features**:
- `aqi_category`: EPA AQI categories
  - Good (0-50)
  - Moderate (51-100)
  - Unhealthy_Sensitive (101-150)
  - Unhealthy (151-200)
  - Very_Unhealthy (201-300)
  - Hazardous (301+)

#### 2.7 Individual Pollutant AQI Values

- `pm25_aqi`, `pm10_aqi`, `no2_aqi`, `o3_aqi`, `so2_aqi`: Individual AQI calculations per pollutant
- `dominant_pollutant`: The pollutant with highest AQI value

**Purpose**: Understand which pollutant is driving overall AQI

#### 2.8 Target Variable Creation (`create_target`)

- `aqi_target_day_1`: AQI 24 hours ahead
- `aqi_target_day_2`: AQI 48 hours ahead
- `aqi_target_day_3`: AQI 72 hours ahead

**Purpose**: Create target variables for multi-day forecasting

#### 2.9 Complete Feature Engineering Pipeline

##### **Main Function** (`engineer_features`)
- **Execution Order**:
  1. Extract time-based features
  2. Compute derived features (lags, rolling stats, ratios, etc.)
  3. Create target variables (optional)
- **Output**: 69 total features (13 original + 56 engineered)
- **Location**: `pipelines/feature_engineering.py`

### 3. Preprocessing Pipeline Flow

```
Raw Data (from API)
    ↓
1. Drop columns with >50% missing values
    ↓
2. Remove duplicate rows (by timestamp)
    ↓
3. Validate data ranges (set invalid to NaN)
    ↓
4. Handle missing values (linear interpolation)
    ↓
5. Cap outliers (99th percentile)
    ↓
Cleaned Data
    ↓
6. Feature Engineering
    ↓
7. AQI Calculation (EPA method)
    ↓
Engineered Features (69 features)
    ↓
Save to CSV & MongoDB
```

### 4. Summary Statistics

#### Preprocessing Methods:
- **7 preprocessing functions** in `DataCleaner` class
- **1 complete pipeline** (`clean_data`) with configurable steps
- **Automatic column dropping** for >50% missing values
- **Multiple missing value strategies** available
- **Outlier detection and capping** implemented

#### Feature Engineering Methods:
- **3 main feature engineering functions**
- **69 total features** generated (13 original + 56 engineered)
- **Time features**: 12 features (6 basic + 6 cyclical)
- **Lag features**: 5 features (1h, 3h, 6h, 12h, 24h)
- **Rolling statistics**: 16 features (4 windows × 4 stats)
- **EMA features**: 3 features
- **Derived features**: 5 features
- **Target variables**: 3 features (for 3-day forecasting)

### 5. Key Design Decisions

1. **Column Dropping Before Processing**: Dropping high-missing columns first prevents wasting computation on unusable features
2. **Interpolation for Missing Values**: Linear interpolation preserves temporal patterns better than mean/median
3. **Outlier Capping vs Removal**: Capping preserves data while reducing extreme values
4. **Cyclical Encoding**: Sin/cos transformations preserve periodicity of time features
5. **Multiple Rolling Windows**: Captures patterns at different time scales (6h, 12h, 24h, 48h)
6. **Essential Column Protection**: Timestamp and metadata columns are never dropped

---

## Model Performance Improvements

### Current Performance (Before Improvements)

- **XGBoost**: R² = 0.462, RMSE = 19.78, MAE = 15.80, MAPE = 20.11%, Overfitting gap = 0.219
- **Ridge Regression**: R² = 0.446, RMSE = 20.07, MAE = 15.97, MAPE = 20.39%, Overfitting gap = 0.017
- **Random Forest**: R² = 0.414, RMSE = 20.63, MAE = 16.56, MAPE = 21.27%, Overfitting gap = 0.164

### Improvements Implemented

#### 1. Enhanced Feature Engineering

**New Features Added:**
- **Interaction Features**: 
  - Temperature × Humidity
  - Temperature × Wind Speed
  - Wind Speed × Humidity
  - Multi-pollutant interactions (PM2.5 × NO2, PM10 × O3, PM2.5 × O3)
  
- **Polynomial Features**:
  - Wind speed squared and cubed
  - Temperature/humidity ratios
  
- **Weather-Derived Features**:
  - Wind direction cyclical encoding (sin/cos)
  - Wind direction categories
  - Pressure change rates and rolling statistics
  - Temperature change rates and rolling statistics
  - Humidity change rates and rolling statistics
  - Wind speed change rates and rolling statistics

**Impact**: These features capture non-linear relationships and temporal patterns that improve model accuracy.

#### 2. Feature Selection

- **SelectKBest**: Automatically selects top 80% of features based on F-statistic correlation with target
- **Threshold**: Applied when more than 20 features are available
- **Benefit**: Reduces noise, improves generalization, and speeds up training

#### 3. Temporal Data Splitting

- **Before**: Random train/test split (not appropriate for time series)
- **After**: Temporal split based on timestamp (earlier data for training, later data for testing)
- **Benefit**: More realistic evaluation and prevents data leakage

#### 4. Advanced Model Training

##### XGBoost Improvements:
- **Early Stopping**: Prevents overfitting by stopping when validation performance stops improving
- **Validation Set**: Uses 20% of training data for early stopping
- **Increased n_estimators**: 200 → 500 (with early stopping)

##### LightGBM Addition:
- **New Model**: Added LightGBM as an additional algorithm option
- **Benefits**: Faster training, often better performance than XGBoost
- **Features**: Early stopping, regularization, feature importance

##### Ensemble Methods:
- **VotingRegressor**: Combines predictions from multiple models (Random Forest, Ridge, XGBoost, LightGBM)
- **Benefit**: Reduces variance and improves generalization

#### 5. Improved Hyperparameter Tuning

- **Ridge Regression**: GridSearchCV with expanded alpha range (0.1 to 100.0)
- **XGBoost**: Early stopping for optimal iteration selection
- **LightGBM**: Comprehensive parameter grid with regularization

#### 6. Better Data Preprocessing

- **Median Imputation**: More robust than mean for skewed distributions
- **StandardScaler**: Feature scaling for better convergence
- **Temporal Ordering**: Ensures data is sorted by timestamp before splitting

### Expected Performance Improvements

After retraining with these improvements, you should see:

1. **Higher R² Score**: Target > 0.50 (from current 0.462)
2. **Lower RMSE**: Target < 18.0 (from current 19.78)
3. **Reduced Overfitting**: Overfitting gap < 0.15 (from current 0.219 for XGBoost)
4. **Better Generalization**: Ensemble model should perform best

### How to Apply Improvements

1. **Install LightGBM** (if not already installed):
   ```bash
   pip install lightgbm>=4.0.0
   ```

2. **Retrain Models**:
   ```bash
   python pipelines/training_pipeline.py
   ```

3. **Monitor Results**:
   - Check logs for feature selection information
   - Compare new metrics with previous performance
   - Ensemble model should show best performance

### Configuration

The improvements are automatically applied when you run the training pipeline. Key settings in `config/config.yaml`:

```yaml
models:
  algorithms:
    - "random_forest"
    - "ridge_regression"
    - "xgboost"
    - "lightgbm"  # New addition
```

### Notes

- Feature selection is applied automatically when > 20 features are available
- Temporal splitting requires timestamp column in features
- Ensemble model is created if at least 2 base models are trained
- All preprocessing steps (scaler, feature selector) are saved with the model for consistent inference

### Next Steps for Further Improvement

1. **More Data**: Collect more historical data (currently ~8,784 records)
2. **External Features**: Add calendar events, holidays, traffic data
3. **Deep Learning**: Consider LSTM/GRU for time series patterns
4. **Hyperparameter Optimization**: Use Optuna or Hyperopt for automated tuning
5. **Feature Engineering**: Domain-specific features based on air quality research

---

## Project Structure

### Directory Organization

This project follows a standard Python project structure with clear separation of concerns between reusable modules and utility scripts.

### `pipelines/` Directory

**Purpose**: Contains reusable, importable Python modules that form the core functionality of the application.

**Characteristics**:
- Contains classes and functions that are imported by other parts of the system
- Modules are designed to be reusable components
- Part of the main application logic
- Can be imported: `from pipelines.data_cleaning import DataCleaner`

**Contents**:
- `data_fetcher.py` - Data fetching classes (OpenMeteoDataFetcher)
- `data_cleaning.py` - Data cleaning classes (DataCleaner)
- `feature_engineering.py` - Feature engineering classes (FeatureEngineer)
- `aqi_calculator.py` - AQI calculation utilities
- `feature_pipeline.py` - Main feature pipeline orchestrator
- `training_pipeline.py` - Model training pipeline
- `mongodb_store.py` - MongoDB storage utilities
- `backfill.py` - Historical data backfill functionality
- `utils.py` - Shared utility functions

**Usage Example**:
```python
from pipelines.data_cleaning import DataCleaner
from pipelines.feature_engineering import FeatureEngineer

cleaner = DataCleaner()
engineer = FeatureEngineer()
```

### `scripts/` Directory

**Purpose**: Contains standalone utility scripts that can be executed directly from the command line.

**Characteristics**:
- Standalone scripts for specific tasks
- Typically run once or on-demand
- Helper utilities for setup, testing, and maintenance
- Can be executed: `python scripts/test_mongodb.py`

**Contents**:
- `test_mongodb.py` - Test MongoDB connection and diagnostics
- `clear_features.py` - Clear features from MongoDB
- `export_to_csv.py` - Export MongoDB data to CSV
- `run_optimized_backfill.py` - Run optimized backfill script
- `run_dashboard.py` - Launch Streamlit dashboard

**Usage Example**:
```bash
python scripts/test_mongodb.py
python scripts/clear_features.py
python scripts/run_optimized_backfill.py --resume
```

### Key Differences

| Aspect | `pipelines/` | `scripts/` |
|--------|--------------|-----------|
| **Purpose** | Reusable modules | Standalone utilities |
| **Importable** | Yes (imported by other modules) | No (run directly) |
| **Reusability** | High (used across project) | Low (specific tasks) |
| **Execution** | Imported and used | Run from command line |
| **Dependencies** | Can depend on each other | Import from pipelines |
| **Examples** | Classes, functions | CLI tools, setup scripts |

### Why This Structure?

#### Benefits:

1. **Separation of Concerns**
   - Core logic (pipelines) is separate from utilities (scripts)
   - Makes codebase easier to navigate and maintain

2. **Reusability**
   - Pipeline modules can be imported anywhere in the project
   - Scripts are self-contained for specific tasks

3. **Modularity**
   - Each pipeline module has a single responsibility
   - Scripts are independent and don't affect core functionality

4. **Testing**
   - Pipeline modules can be unit tested easily
   - Scripts can be tested as integration tests

5. **CI/CD Integration**
   - Pipeline modules are used in automated workflows
   - Scripts can be run manually or in maintenance tasks

### Import Patterns

#### Scripts Import from Pipelines:
```python
# scripts/test_mongodb.py
from pipelines.mongodb_store import MongoDBStore
from config.settings import config
```

#### Pipelines Import from Each Other:
```python
# pipelines/feature_pipeline.py
from pipelines.data_fetcher import OpenMeteoDataFetcher
from pipelines.data_cleaning import DataCleaner
from pipelines.feature_engineering import FeatureEngineer
```

#### Application Code Imports from Pipelines:
```python
# api/main.py
from pipelines.data_fetcher import OpenMeteoDataFetcher
from pipelines.training_pipeline import ModelTrainer
```

### When to Add to Each Directory

#### Add to `pipelines/` when:
- Creating reusable classes or functions
- Building core application functionality
- Code that will be imported by multiple parts of the system
- Data processing, model training, feature engineering logic

#### Add to `scripts/` when:
- Creating one-off utility scripts
- Setup or configuration scripts
- Diagnostic or testing scripts
- Maintenance or data migration scripts
- Scripts that users run manually

### Current Project Flow

```
User runs script (scripts/)
    ↓
Script imports from pipelines/
    ↓
Pipeline modules process data
    ↓
Results stored in MongoDB/CSV
```

**Example Flow**:
1. User runs: `python scripts/run_optimized_backfill.py`
2. Script imports: `from pipelines.backfill import backfill_historical_data`
3. Backfill uses: `from pipelines.data_fetcher import OpenMeteoDataFetcher`
4. Data is processed and stored

### Best Practices

1. **Keep pipelines modular**: Each module should have a single, clear purpose
2. **Keep scripts simple**: Scripts should be focused on one task
3. **Avoid circular dependencies**: Scripts import from pipelines, not vice versa
4. **Document both**: Both directories should have clear documentation
5. **Test pipelines**: Pipeline modules should be unit tested
6. **Scripts are utilities**: Scripts are for convenience, not core logic

### Summary

- **`pipelines/`** = Reusable, importable core modules (the "library")
- **`scripts/`** = Standalone utility scripts (the "tools")

This structure is common in Python projects and follows best practices for maintainability and organization.
