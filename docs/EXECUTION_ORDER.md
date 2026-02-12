# Project Execution Order Guide

This document provides the step-by-step order to execute all files to get the AQI prediction project running.

---

## Prerequisites

Before starting, ensure you have:
- Python 3.12+ installed
- Virtual environment created and activated
- Dependencies installed: `pip install -r requirements.txt`

## CI/CD Automation

This project includes GitHub Actions workflows for automated execution:
- **Feature Pipeline**: Runs automatically every hour
- **Training Pipeline**: Runs automatically daily at 2 AM UTC

For setup instructions, see [CI_CD_SETUP.md](CI_CD_SETUP.md).

---

## Phase 1: Initial Setup and Configuration

### Step 1.1: Environment Setup
**File**: `.env` (create manually or use script)

Create `.env` file in project root with:
```
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/database
OPENMETEO_API_KEY=your_key_here (optional)
```

### Step 1.2: Configuration Setup
**File**: `config/config.yaml`

Verify city coordinates and settings are correct (already configured for Karachi).

### Step 1.3: Test MongoDB Connection
**File**: `scripts/test_mongodb.py`

```bash
python scripts/test_mongodb.py
```

**Purpose**: Verify MongoDB connection is working
**Expected Output**: Connection successful message

**If connection fails**:
- Run diagnostics: `python scripts/test_mongodb.py diagnose`
- Check IP whitelist in MongoDB Atlas
- Verify connection string encoding

---

## Phase 2: Data Collection (Historical Backfill)

### Step 2.1: Backfill Historical Data
**START HERE**: `pipelines/backfill.py`

**Direct execution** (recommended):
```bash
python pipelines/backfill.py
```

This will automatically:
- Use yesterday as end date (excludes current day)
- Fetch last 365 days of historical data
- Process in batches of 30 days

**Alternative - Custom date range**:
```bash
python pipelines/backfill.py --start-date 2025-01-16 --end-date 2026-01-16 --batch-days 30
```

**Purpose**: 
- Fetch historical weather data (1 year)
- Estimate historical AQI values
- Store raw data in `data/raw/raw_data.csv`
- Store features in `data/features/features.csv`
- Store in MongoDB (with automatic duplicate prevention)

**Duplicate Prevention**: 
- The pipeline automatically checks for existing timestamps before inserting
- Running the backfill multiple times will NOT create duplicates
- Only new records (with new timestamps) will be inserted

**Expected Output**: 
- ~8,760 records (365 days × 24 hours)
- Raw data CSV: `data/raw/raw_data.csv`
- Features CSV: `data/features/features.csv`
- Data in MongoDB collection `aqi_features`

**Time**: ~10-15 minutes

**Note**: You can safely run this multiple times - duplicates are automatically prevented

---

## Phase 3: Verify Data Collection

### Step 3.1: Check Data Files
**Files**: Check CSV files manually or use script

```bash
# Check raw data
python -c "import pandas as pd; df = pd.read_csv('data/raw/raw_data.csv'); print(f'Raw records: {len(df)}'); print(f'Date range: {df[\"timestamp\"].min()} to {df[\"timestamp\"].max()}')"

# Check features
python -c "import pandas as pd; df = pd.read_csv('data/features/features.csv'); print(f'Feature records: {len(df)}'); print(f'Features: {len(df.columns)}')"
```

**Expected**: 
- Raw data: ~8,760 records 
- Features: ~8,760 records, 69 columns

### Step 3.2: Export MongoDB Data (Optional)
**File**: `scripts/export_to_csv.py`

```bash
python scripts/export_to_csv.py
```

**Purpose**: Export all MongoDB collections to CSV for backup
**Output**: CSV files in `data/features/` directory

---

## Phase 4: Model Training

### Step 4.1: Train Models
**File**: `pipelines/training_pipeline.py`

```bash
python pipelines/training_pipeline.py
```

**Purpose**:
- Load features from MongoDB
- Train multiple models (Random Forest, Ridge, XGBoost, Neural Network)
- Evaluate models using cross-validation
- Save best model to MongoDB and local filesystem

**Expected Output**:
- Model saved to `models/random_forest/model.pkl` (or best model)
- Model metadata in MongoDB collection `aqi_models`
- Training metrics logged

**Time**: ~5-10 minutes (depending on data size)

**If training fails with "constant target" error**:
- Check if AQI values are varied (not all same value)
- Re-run backfill if needed
- Check data quality

---

## Phase 5: API Service (Prediction Endpoint)

### Step 5.1: Start FastAPI Service
**File**: `api/main.py`

```bash
# Option 1: Direct Python
python api/main.py

# Option 2: Using uvicorn
uvicorn api.main:app --host 0.0.0.0 --port 8000

# Option 3: Using Docker (if configured)
docker build -t aqi-api -f api/Dockerfile .
docker run -p 8000:8000 aqi-api
```

**Purpose**: 
- Start REST API service for predictions
- Endpoints:
  - `GET /health` - Health check
  - `GET /predict` - Get AQI prediction
  - `GET /predict/forecast` - Get 3-day forecast

**Expected Output**: 
- Server shows: `Uvicorn running on http://0.0.0.0:8000` (bind address)
- **Access via**: `http://localhost:8000` or `http://127.0.0.1:8000` in your browser
- API documentation at `http://localhost:8000/docs`

**Important**: `0.0.0.0` is the bind address (listens on all network interfaces). Use `localhost` or `127.0.0.1` to access in your browser.

**Keep running**: Leave this terminal open

---

## Phase 6: Dashboard (Web Interface)

### Step 6.1: Start Streamlit Dashboard
**File**: `app/dashboard.py` or `scripts/run_dashboard.py`

**Option A - Using script**:
```bash
python scripts/run_dashboard.py
```

**Option B - Direct Streamlit**:
```bash
streamlit run app/dashboard.py
```

**Note**: Auto-reload is enabled by default via `.streamlit/config.toml`. The dashboard will automatically refresh when you save changes to the code. No need to manually restart the server.

**Purpose**: 
- Interactive web dashboard
- Display current AQI
- Show 3-day forecast
- Visualize trends

**How to Get Predictions**:
1. Open dashboard in browser: `http://localhost:8501`
2. Adjust "Forecast Days" slider (1-7 days, default: 3)
3. Click "Get Predictions" button
4. Dashboard will:
   - Fetch current air quality data from Open-Meteo API
   - Fetch current weather forecast
   - Generate features from the data
   - Use trained ML model to predict AQI for next N days
   - Display results in charts and tables

**Expected Output**: 
- Streamlit shows: `URL: http://0.0.0.0:8501` (bind address)
- **Access via**: `http://localhost:8501` or `http://127.0.0.1:8501` in your browser
- Dashboard shows AQI predictions and charts after clicking "Get Predictions"

**Note**: 
- If you see `0.0.0.0:8501` in the terminal, use `localhost:8501` in your browser instead
- City location is configured in `config/config.yaml` (currently: Karachi)
- No manual input needed - predictions are generated automatically from real-time API data

**Keep running**: Leave this terminal open

---

## Phase 7: Automated Pipelines (CI/CD)

### Step 7.1: Feature Pipeline (Hourly)
**File**: `.github/workflows/feature_pipeline.yml`

**Automated**: Runs every hour via GitHub Actions

**Manual execution**:
```bash
python pipelines/feature_pipeline.py
```

**Purpose**:
- Fetch current air quality and weather data
- Clean and engineer features
- Store in MongoDB and CSV

### Step 7.2: Training Pipeline (Daily)
**File**: `.github/workflows/training_pipeline.yml`

**Automated**: Runs daily via GitHub Actions

**Manual execution**:
```bash
python pipelines/training_pipeline.py
```

**Purpose**:
- Retrain models with new data
- Update model if performance improves

---

## Complete Execution Order Summary

### First-Time Setup:
```
1. Create .env file with MongoDB URI
2. python scripts/test_mongodb.py                    # Test connection
3. python scripts/run_optimized_backfill.py --resume # Backfill data
4. python pipelines/training_pipeline.py              # Train models
5. python api/main.py                                 # Start API (Terminal 1)
6. streamlit run app/dashboard.py                     # Start dashboard (Terminal 2)
```

### Daily Operations:
```
1. Feature pipeline runs automatically (hourly) OR
   python pipelines/feature_pipeline.py               # Manual run
2. Training pipeline runs automatically (daily) OR
   python pipelines/training_pipeline.py              # Manual run
3. API and Dashboard stay running
```

### Maintenance Tasks:
```
# Clear old data
python scripts/clear_features.py

# Export data
python scripts/export_to_csv.py

# Test connection
python scripts/test_mongodb.py diagnose
```

---

## Quick Start (All-in-One)

For a complete setup from scratch:

```bash
# 1. Setup
python scripts/test_mongodb.py

# 2. Backfill (if no data exists)
python scripts/run_optimized_backfill.py --resume

# 3. Train models
python pipelines/training_pipeline.py

# 4. Start services (in separate terminals)
# Terminal 1:
python api/main.py

# Terminal 2:
streamlit run app/dashboard.py
```

---

## File Execution Dependencies

### Dependency Tree:
```
test_mongodb.py
    └─> Uses: pipelines/mongodb_store.py

run_optimized_backfill.py
    └─> Uses: pipelines/backfill.py
        └─> Uses: pipelines/data_fetcher.py
        └─> Uses: pipelines/data_cleaning.py
        └─> Uses: pipelines/feature_engineering.py
        └─> Uses: pipelines/aqi_calculator.py
        └─> Uses: pipelines/mongodb_store.py

training_pipeline.py
    └─> Uses: pipelines/mongodb_store.py
    └─> Uses: pipelines/data_cleaning.py

feature_pipeline.py
    └─> Uses: pipelines/data_fetcher.py
    └─> Uses: pipelines/data_cleaning.py
    └─> Uses: pipelines/feature_engineering.py
    └─> Uses: pipelines/aqi_calculator.py
    └─> Uses: pipelines/mongodb_store.py

api/main.py
    └─> Uses: pipelines/data_fetcher.py
    └─> Uses: pipelines/training_pipeline.py
    └─> Uses: pipelines/mongodb_store.py

app/dashboard.py
    └─> Uses: api/main.py (via HTTP requests)
```

---

## Troubleshooting Execution Order

### If MongoDB connection fails:
1. Run `python scripts/test_mongodb.py diagnose`
2. Check `.env` file exists and has correct `MONGODB_URI`
3. Verify IP whitelist in MongoDB Atlas

### If backfill fails:
1. Check internet connection
2. Verify Open-Meteo API is accessible
3. Check date ranges (should not include current day)
4. Review logs for specific errors

### If training fails:
1. Verify data exists: Check `data/features/features.csv`
2. Check for constant AQI values (overfitting issue)
3. Ensure sufficient data (at least 1000+ records recommended)
4. Review training logs

### If API fails to start:
1. Verify model exists: Check `models/` directory
2. Check MongoDB connection
3. Verify port 8000 is not in use
4. Check API logs

### If dashboard fails:
1. Verify API is running on port 8000
2. Check Streamlit installation: `pip install streamlit`
3. Verify dashboard can reach API endpoint

---

## Execution Checklist

- [ ] Environment variables configured (`.env`)
- [ ] MongoDB connection tested
- [ ] Historical data backfilled
- [ ] Data verified in CSV files
- [ ] Models trained successfully
- [ ] API service running
- [ ] Dashboard accessible
- [ ] Automated pipelines configured (GitHub Actions)

---

## Notes

1. **Terminal Management**: API and Dashboard need separate terminals (or run in background)
2. **Data Requirements**: Minimum 1000+ records recommended for training
3. **Time Estimates**: 
   - Backfill: 10-15 minutes (optimized)
   - Training: 5-10 minutes
   - API/Dashboard: Start immediately
4. **Automation**: Once set up, GitHub Actions handle hourly/daily pipelines
5. **Data Updates**: Feature pipeline runs hourly to keep data current

---

## Next Steps After Setup

1. Monitor dashboard for predictions
2. Check GitHub Actions for pipeline status
3. Review model performance metrics
4. Adjust configuration as needed
5. Set up alerts/notifications (optional)
