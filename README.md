# 10pearlsAQI - Air Quality Index Prediction System

**Live Application**: https://10pearls-aqi-predictor.streamlit.app/  
**API Endpoint**: https://10pearlsaqi-production.up.railway.app/  
**Submission Portal**: https://shine.10pearls.com/candidate/submissions

![Project Dashboard](assets/image.png)

---

## 📋 Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Key Features](#key-features)
4. [Technology Stack](#technology-stack)
5. [Live Deployment](#live-deployment)
6. [Installation & Setup](#installation--setup)
7. [Project Structure](#project-structure)
8. [Usage Guide](#usage-guide)
9. [Machine Learning Models](#machine-learning-models)
10. [CI/CD Pipeline](#cicd-pipeline)
11. [API Documentation](#api-documentation)
12. [Performance Metrics](#performance-metrics)
13. [Configuration](#configuration)
14. [Troubleshooting](#troubleshooting)
15. [Project Timeline](#project-timeline)

---

## 🎯 Project Overview

**10pearlsAQI** is an end-to-end Air Quality Index (AQI) prediction system that forecasts air quality for Karachi, Pakistan, for the next 1-7 days using machine learning models. The system features automated data collection, feature engineering, model training, and a production-ready web application.

### Objectives Achieved

✅ **Automated Data Pipeline**: Hourly data collection from Open-Meteo API  
✅ **Feature Engineering**: 69 engineered features including temporal, cyclical, and derived features  
✅ **Multiple ML Models**: Random Forest, Ridge Regression, XGBoost, LightGBM, and Ensemble  
✅ **Production API**: FastAPI service deployed on Railway  
✅ **Interactive Dashboard**: Streamlit web interface deployed on Streamlit Cloud  
✅ **CI/CD Automation**: GitHub Actions for automated hourly/daily pipelines  
✅ **Cloud Storage**: MongoDB Atlas for scalable feature and model storage  
✅ **Model Registry**: Automated model versioning and best model selection

### Key Achievements

- **Best Model Performance**: Ensemble model with RMSE: 20.92, R²: 0.231, MAE: 16.91
- **Data Coverage**: 8,787 hourly records covering 365+ days
- **Feature Count**: 69 engineered features per record
- **Prediction Horizon**: 1-7 days ahead
- **Automation**: Fully automated hourly feature updates and daily model retraining
- **Production Ready**: Live deployment with real-time predictions

---

## 🏗️ System Architecture

### High-Level Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Open-Meteo API │────▶│  Feature Pipeline│────▶│  MongoDB Atlas  │
│  (Data Source)  │     │  (Hourly)        │     │  (Feature Store)│
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit      │────▶│  FastAPI Service │────▶│  Model Registry │
│  Dashboard      │     │  (Railway)        │     │  (MongoDB)      │
│  (Streamlit     │     │                   │     │                 │
│   Cloud)        │     └──────────────────┘     └─────────────────┘
└─────────────────┘
         │
         ▼
┌─────────────────┐
│  GitHub Actions │
│  (CI/CD)         │
│  - Hourly:       │
│    Features      │
│  - Daily:        │
│    Training      │
└─────────────────┘
```

### Component Details

1. **Data Collection Layer**
   - Open-Meteo API for air quality and weather data
   - Automated hourly data fetching
   - Historical data backfill capability

2. **Feature Engineering Layer**
   - Temporal features (hour, day, month, season)
   - Cyclical encoding (sin/cos transformations)
   - Derived features (ratios, interactions, rolling statistics)
   - AQI calculation using EPA standards

3. **Machine Learning Layer**
   - Multiple model training (Random Forest, Ridge, XGBoost, LightGBM)
   - Ensemble model (VotingRegressor)
   - Automated best model selection
   - Model versioning and storage

4. **API Layer**
   - FastAPI RESTful service
   - Real-time prediction endpoints
   - Model management endpoints
   - Health check endpoints

5. **Presentation Layer**
   - Streamlit interactive dashboard
   - Real-time AQI visualization
   - Multi-day forecast display
   - Model performance metrics

6. **Automation Layer**
   - GitHub Actions workflows
   - Hourly feature pipeline
   - Daily training pipeline
   - Automated error notifications

---

## ✨ Key Features

### 1. Automated Data Pipeline
- **Hourly Data Collection**: Automatically fetches latest air quality and weather data
- **Feature Engineering**: Creates 69 features including:
  - Time-based features (hour, day, month, season)
  - Cyclical encoding (sin/cos for periodic patterns)
  - Rolling statistics (mean, std, min, max over 6h, 12h, 24h, 48h windows)
  - Exponential moving averages
  - Pollutant ratios and interactions
  - Weather indices (heat index, comfort index)
  - Wind direction categorization
- **Data Storage**: Stores features in MongoDB Atlas for scalable access

### 2. Machine Learning Models
- **Multiple Algorithms**: 
  - Random Forest (baseline)
  - Ridge Regression (linear baseline)
  - XGBoost (gradient boosting)
  - LightGBM (fast gradient boosting)
  - Ensemble (VotingRegressor combining all models)
- **Automated Selection**: Best model selected based on lowest RMSE
- **Model Registry**: Versioned model storage in MongoDB
- **Performance Tracking**: RMSE, MAE, R², MAPE metrics

### 3. Production API
- **FastAPI Service**: RESTful API for predictions
- **Endpoints**:
  - `GET /health` - Health check
  - `GET /models` - List available models
  - `POST /predict` - Get AQI predictions (1-7 days)
- **Deployment**: Railway cloud platform
- **Auto-scaling**: Handles cold starts and traffic spikes

### 4. Interactive Dashboard
- **Real-time Predictions**: Fetches current data and generates forecasts
- **Visualizations**: Interactive Plotly charts for AQI trends
- **Multi-day Forecast**: 1-7 day predictions with confidence intervals
- **Model Information**: Displays best model and performance metrics
- **Alert System**: Warnings for unhealthy AQI levels
- **Deployment**: Streamlit Cloud with auto-reload

### 5. CI/CD Automation
- **GitHub Actions Workflows**:
  - **Feature Pipeline**: Runs every hour at minute 0
  - **Training Pipeline**: Runs daily at 2 AM UTC
- **Automated Notifications**: Creates GitHub issues on failure
- **Artifact Storage**: Uploads features and models as artifacts
- **Error Handling**: Comprehensive logging and error recovery

### 6. Cloud Integration
- **MongoDB Atlas**: Cloud database for features and models
- **Railway**: FastAPI backend hosting
- **Streamlit Cloud**: Dashboard hosting
- **GitHub Actions**: CI/CD automation

---

## 🛠️ Technology Stack

### Backend
- **Python 3.12**: Core programming language
- **FastAPI 0.128.0**: RESTful API framework
- **Uvicorn**: ASGI server
- **Pydantic**: Data validation

### Machine Learning
- **scikit-learn 1.8.0**: Random Forest, Ridge Regression, feature selection
- **XGBoost 3.1.3**: Gradient boosting
- **LightGBM 4.6.0**: Fast gradient boosting
- **NumPy 2.4.1**: Numerical computations
- **Pandas 2.3.3**: Data manipulation

### Data Storage
- **MongoDB Atlas**: Cloud database
- **PyMongo 4.16.0**: MongoDB driver
- **Motor 3.7.1**: Async MongoDB driver

### Frontend
- **Streamlit 1.53.0**: Web dashboard framework
- **Plotly 6.5.2**: Interactive visualizations
- **Pandas**: Data display

### DevOps
- **GitHub Actions**: CI/CD automation
- **Railway**: API hosting
- **Streamlit Cloud**: Dashboard hosting

### Data Sources
- **Open-Meteo API**: Air quality and weather data (free tier)

---

## 🌐 Live Deployment

### Dashboard
**URL**: https://10pearls-aqi-predictor.streamlit.app/

**Features**:
- Real-time AQI predictions
- Interactive visualizations
- Model performance metrics
- Multi-day forecasts

### API
**Base URL**: https://10pearlsaqi-production.up.railway.app/

**Endpoints**:
- Health Check: `GET /health`
- List Models: `GET /models`
- Predictions: `POST /predict`

**Example API Call**:
```bash
curl -X POST https://10pearlsaqi-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3}'
```

---

## 📦 Installation & Setup

### Prerequisites

- Python 3.10 or higher
- MongoDB Atlas account (free tier available)
- Git

### Step 1: Clone Repository

```bash
git clone <repository-url>
cd 10pearlsAQI
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables

Create `.env` file in project root:

```env
# MongoDB Connection (Required)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/aqi_predictor?retryWrites=true&w=majority

# API Configuration (Optional - for local development)
FASTAPI_URL=http://localhost:8000
```

### Step 5: Configure City Settings

Edit `config/config.yaml`:

```yaml
city:
  name: "Karachi"
  latitude: 24.8608
  longitude: 67.0104
  timezone: "Asia/Karachi"
```

### Step 6: Test MongoDB Connection

```bash
python scripts/test_mongodb.py
```

### Step 7: Backfill Historical Data (Optional)

```bash
python scripts/run_optimized_backfill.py --resume
```

### Step 8: Train Models

```bash
python pipelines/training_pipeline.py
```

### Step 9: Start Services

**Terminal 1 - FastAPI**:
```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

**Terminal 2 - Streamlit Dashboard**:
```bash
streamlit run app/dashboard.py
```

Access dashboard at: http://localhost:8501

---

## 📁 Project Structure

```
10pearlsAQI/
├── api/                          # FastAPI service
│   ├── main.py                  # API endpoints
│   ├── requirements.txt         # API dependencies
│   └── Dockerfile               # Docker configuration
│
├── app/                          # Streamlit dashboard
│   ├── dashboard.py             # Dashboard application
│   └── __init__.py
│
├── pipelines/                    # Core ML pipelines
│   ├── data_fetcher.py          # API data fetching
│   ├── data_cleaning.py         # Data preprocessing
│   ├── feature_engineering.py   # Feature creation (69 features)
│   ├── feature_pipeline.py      # Main feature pipeline
│   ├── training_pipeline.py     # Model training
│   ├── mongodb_store.py         # MongoDB operations
│   ├── aqi_calculator.py        # EPA AQI calculation
│   └── utils.py                 # Utility functions
│
├── config/                       # Configuration
│   ├── config.yaml              # Main configuration
│   ├── settings.py               # Settings loader
│   └── env.example              # Environment template
│
├── scripts/                      # Utility scripts
│   ├── test_mongodb.py          # MongoDB connection test
│   ├── run_optimized_backfill.py # Historical data backfill
│   └── check_data_quality.py    # Data quality checks
│
├── docs/                         # Documentation
│   ├── SETUP_GUIDE.md           # Complete setup guide
│   ├── DEPLOYMENT_GUIDE.md      # Deployment instructions
│   ├── TECHNICAL_DOCS.md        # Technical documentation
│   └── SUPERVISOR_REPORT.md     # Project report
│
├── .github/workflows/            # CI/CD pipelines
│   ├── feature_pipeline.yml     # Hourly feature pipeline
│   └── training_pipeline.yml     # Daily training pipeline
│
├── .streamlit/                   # Streamlit config
│   ├── config.toml              # Streamlit settings
│   └── secrets.toml.example     # Secrets template
│
├── data/                         # Local data storage
│   ├── features/                # Feature CSV files
│   └── raw/                     # Raw data CSV files
│
├── models/                       # Local model backup
│   ├── random_forest/
│   ├── xgboost/
│   └── ensemble/
│
├── requirements.txt              # Python dependencies
├── railway.toml                  # Railway deployment config
├── render.yaml                   # Render deployment config
└── README.md                     # This file
```

---

## 🚀 Usage Guide

### Running Feature Pipeline

Fetch current data and generate features:

```bash
python pipelines/feature_pipeline.py
```

This will:
1. Fetch current air quality and weather data
2. Clean and preprocess the data
3. Engineer 69 features
4. Store features in MongoDB

### Training Models

Train all ML models:

```bash
python pipelines/training_pipeline.py
```

This will:
1. Load features from MongoDB
2. Split data into train/test sets
3. Train multiple models (RF, Ridge, XGBoost, LightGBM)
4. Create ensemble model
5. Evaluate and store best model in MongoDB

### Using the API

**Health Check**:
```bash
curl https://10pearlsaqi-production.up.railway.app/health
```

**List Models**:
```bash
curl https://10pearlsaqi-production.up.railway.app/models
```

**Get Predictions**:
```bash
curl -X POST https://10pearlsaqi-production.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3}'
```

### Using the Dashboard

1. Open https://10pearls-aqi-predictor.streamlit.app/
2. Adjust "Forecast Days" slider (1-7 days)
3. Click "Get Predictions"
4. View interactive charts and forecasts

---

## 🤖 Machine Learning Models

### Model Performance

| Model | RMSE | MAE | R² | MAPE | Overfitting Gap |
|-------|------|-----|----|----|-----------------|
| **Ensemble** | **20.92** | **16.91** | **0.231** | **18.30%** | 0.466 |
| LightGBM | 21.10 | 17.11 | 0.217 | 18.53% | 0.386 |
| XGBoost | 21.34 | 17.21 | 0.199 | 18.79% | 0.612 |
| Random Forest | 21.31 | 17.21 | 0.202 | 18.66% | 0.389 |
| Ridge Regression | 22.50 | 17.59 | 0.110 | 18.61% | 0.386 |

**Best Model**: Ensemble (VotingRegressor combining all models)

### Model Selection

The system automatically selects the model with the **lowest RMSE** for predictions. Invalid models (RMSE = 0) are filtered out.

### Feature Engineering

**69 Features** per record:
- **Temporal Features** (12): hour, day, month, season, cyclical encodings
- **Pollutant Features** (15): PM2.5, PM10, O3, NO2, CO, SO2, AQI
- **Weather Features** (12): temperature, humidity, pressure, wind speed/direction
- **Derived Features** (15): ratios, interactions, rolling statistics
- **Time-based Features** (15): lag features, change rates, moving averages

### Training Process

1. **Data Split**: Temporal split (80% train, 20% test)
2. **Feature Selection**: SelectKBest (top 80% features)
3. **Preprocessing**: Median imputation, StandardScaler
4. **Hyperparameter Tuning**: 
   - Ridge: GridSearchCV (alpha: 0.1 to 100.0)
   - XGBoost/LightGBM: Early stopping with validation set
5. **Evaluation**: RMSE, MAE, R², MAPE metrics
6. **Model Storage**: Best model saved to MongoDB Model Registry

---

## 🔄 CI/CD Pipeline

### GitHub Actions Workflows

#### 1. Feature Pipeline (Hourly)

**File**: `.github/workflows/feature_pipeline.yml`

**Schedule**: Every hour at minute 0 (`0 * * * *`)

**Steps**:
1. Checkout code
2. Set up Python 3.12
3. Clean disk space
4. Install dependencies (minimal, CPU-only)
5. Run feature pipeline
6. Upload artifacts (CSV files, logs)
7. Create GitHub issue on failure

**Artifacts**: Feature CSV files, logs (retention: 7 days)

#### 2. Training Pipeline (Daily)

**File**: `.github/workflows/training_pipeline.yml`

**Schedule**: Daily at 2 AM UTC (`0 2 * * *`)

**Steps**:
1. Checkout code
2. Set up Python 3.12
3. Clean disk space
4. Install dependencies (ML libraries: scikit-learn, XGBoost, LightGBM)
5. Run training pipeline
6. Upload artifacts (trained models, logs)
7. Create GitHub issue on failure

**Artifacts**: Trained models (.pkl, .json), logs (retention: 30 days)

### Monitoring

- **Workflow Runs**: View in GitHub Actions tab
- **Logs**: Available in each workflow run
- **Notifications**: GitHub issues created on failure
- **Artifacts**: Downloadable from workflow runs

See `docs/WORKFLOW_MONITORING.md` for detailed monitoring guide.

---

## 📡 API Documentation

### Base URL

```
https://10pearlsaqi-production.up.railway.app
```

### Endpoints

#### 1. Health Check

**GET** `/health`

Check API health and MongoDB connection.

**Response**:
```json
{
  "status": "healthy",
  "mongodb": "connected",
  "timestamp": "2026-01-17T19:00:00",
  "service": "AQI Prediction API"
}
```

#### 2. List Models

**GET** `/models`

Get list of available trained models.

**Response**:
```json
{
  "models": [
    {
      "name": "xgboost",
      "trained_at": "2026-01-17T02:00:00",
      "metrics": {
        "rmse": 19.78,
        "mae": 16.21,
        "r2": 0.462,
        "mape": 18.30
      }
    }
  ]
}
```

#### 3. Predict AQI

**POST** `/predict`

Get AQI predictions for next N days.

**Request Body**:
```json
{
  "forecast_days": 3,
  "latitude": 24.8608,
  "longitude": 67.0104
}
```

**Response**:
```json
{
  "predictions": [
    {
      "date": "2026-01-18",
      "predicted_aqi": 85.5,
      "category": "Moderate"
    }
  ],
  "current_aqi": 82.3,
  "model_name": "xgboost",
  "model_metrics": {
    "rmse": 19.78,
    "r2": 0.462
  },
  "generated_at": "2026-01-17T19:00:00"
}
```

---

## 📊 Performance Metrics

### Model Performance Summary

- **Best Model**: Ensemble (VotingRegressor)
- **RMSE**: 20.92 AQI units
- **MAE**: 16.91 AQI units
- **R² Score**: 0.231 (23.1% variance explained)
- **MAPE**: 18.30% (mean absolute percentage error)

### Data Statistics

- **Total Records**: 8,787 hourly records
- **Time Period**: 365+ days
- **Features**: 69 per record
- **Training Set**: 6,976 samples (80%)
- **Test Set**: 1,745 samples (20%)

### System Performance

- **API Response Time**: < 5 seconds (after cold start)
- **Dashboard Load Time**: < 3 seconds
- **Pipeline Execution**: 
  - Feature Pipeline: ~30 seconds
  - Training Pipeline: ~5-10 minutes
- **Cold Start**: 30-60 seconds (Railway)

---

## ⚙️ Configuration

### Environment Variables

**Required**:
- `MONGODB_URI`: MongoDB Atlas connection string

**Optional**:
- `FASTAPI_URL`: FastAPI service URL (default: http://localhost:8000)
- `ALLOWED_ORIGINS`: CORS allowed origins (default: *)

### Configuration File

Edit `config/config.yaml`:

```yaml
city:
  name: "Karachi"
  latitude: 24.8608
  longitude: 67.0104
  timezone: "Asia/Karachi"

models:
  algorithms:
    - "random_forest"
    - "ridge_regression"
    - "xgboost"
    - "lightgbm"
```

### Streamlit Secrets (for Streamlit Cloud)

In Streamlit Cloud dashboard → Settings → Secrets:

```toml
FASTAPI_URL = "https://10pearlsaqi-production.up.railway.app"
MONGODB_URI = "mongodb+srv://..."
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. MongoDB Connection Errors

**Problem**: `Connection refused` or `Name or service not known`

**Solution**:
- Verify `MONGODB_URI` is set correctly
- Check MongoDB Atlas network access (allow 0.0.0.0/0 or specific IPs)
- Ensure credentials are correct

#### 2. API Timeout Errors

**Problem**: `Read timed out` or `502 Bad Gateway`

**Solution**:
- Railway cold starts take 30-60 seconds
- Dashboard automatically retries with exponential backoff
- Check Railway logs for errors

#### 3. Model Not Found

**Problem**: `No trained model found`

**Solution**:
- Run training pipeline: `python pipelines/training_pipeline.py`
- Verify models exist in MongoDB
- Check model collection name in config

#### 4. CI/CD Pipeline Failures

**Problem**: Workflow fails with "No space left on device"

**Solution**:
- Workflows use minimal dependencies (CPU-only)
- Disk space cleanup is automatic
- Check workflow logs for specific errors

See `docs/RAILWAY_TROUBLESHOOTING.md` and `docs/DEPLOYMENT_GUIDE.md` for detailed troubleshooting.

---

## 📅 Project Timeline

**Due Date**: February 13, 2026 at 6:20 PM

**Project Phases**:
1. ✅ Data Collection & API Integration
2. ✅ Feature Engineering Pipeline
3. ✅ Machine Learning Model Development
4. ✅ API Development (FastAPI)
5. ✅ Dashboard Development (Streamlit)
6. ✅ CI/CD Pipeline Setup
7. ✅ Cloud Deployment
8. ✅ Documentation & Testing

**Current Status**: ✅ **Production Ready**

---

## 📚 Additional Documentation

- **Setup Guide**: `docs/SETUP_GUIDE.md` - Complete setup instructions
- **Deployment Guide**: `docs/DEPLOYMENT_GUIDE.md` - Cloud deployment steps
- **Technical Docs**: `docs/TECHNICAL_DOCS.md` - Technical details
- **Supervisor Report**: `docs/SUPERVISOR_REPORT.md` - Comprehensive project report
- **Workflow Monitoring**: `docs/WORKFLOW_MONITORING.md` - CI/CD monitoring guide

---

## 🎓 Project Requirements Met

✅ **Data Collection**: Automated hourly data fetching from Open-Meteo API  
✅ **Data Preprocessing**: Comprehensive cleaning and validation  
✅ **Feature Selection**: 69 engineered features with automated selection  
✅ **MongoDB Integration**: Features and models stored in MongoDB Atlas  
✅ **ML Models**: Multiple models (RF, Ridge, XGBoost, LightGBM, Ensemble)  
✅ **Model Evaluation**: RMSE, MAE, R², MAPE metrics  
✅ **Results Analysis**: Performance comparison and best model selection  
✅ **Frontend**: Interactive Streamlit dashboard  
✅ **CI/CD Pipeline**: GitHub Actions with hourly/daily automation  
✅ **Cloud Deployment**: Production-ready deployment on Railway and Streamlit Cloud

---

## 🔗 Resources

- **Project Resources**: https://drive.google.com/file/d/1HPf17hvqI6icNTjRPkPuydkV1ub_lxO5/view?usp=sharing
- **GitHub Actions Tutorial**: https://www.youtube.com/playlist?list=PLiO7XHcmTsleVSRaY7doSfZryYWMkMOxB
- **Open-Meteo API**: https://open-meteo.com/
- **MongoDB Atlas**: https://www.mongodb.com/cloud/atlas

---

## 👤 Author

**Sagar Chhabriya**  
**Project**: 10pearlsAQI - Air Quality Index Prediction System  
**Submission Portal**: https://shine.10pearls.com/candidate/submissions

---

## 📝 License

This project is part of the 10Pearls Shine Internship Program.

---

## 🙏 Acknowledgments

- Open-Meteo for free air quality and weather data API
- MongoDB Atlas for cloud database hosting
- Railway for API hosting platform
- Streamlit Cloud for dashboard hosting
- GitHub Actions for CI/CD automation

---

**Last Updated**: January 17, 2026  
**Version**: 1.0.0  
**Status**: Production Ready ✅
