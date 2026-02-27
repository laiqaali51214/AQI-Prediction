## AQI Predictor – Air Quality Index Prediction System

**Live Dashboard**: https://rawalpindi-aqi-prediction.streamlit.app/

**API Endpoint (Railway)**: https://aqi-prediction-production-5eee.up.railway.app/

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


---

## 🎯 Project Overview

**AQI Predictor** is an end‑to‑end Air Quality Index (AQI) prediction system that forecasts air quality for **Rawalpindi, Pakistan** using machine learning. It includes automated feature generation, a MongoDB‑backed feature store and model registry, a FastAPI backend for predictions, and a Streamlit dashboard for interactive visualisation.

### Objectives

- **Automated feature pipeline** that regularly ingests and processes air quality and weather data from Open‑Meteo.
- **Feature engineering** to build a rich tabular dataset suitable for supervised learning.
- **Multiple ML models** (Random Forest, Ridge Regression, XGBoost, LightGBM, Ensemble).
- **Production‑ready API** exposing health, model listing, and prediction endpoints.
- **Interactive dashboard** in Streamlit consuming the FastAPI service.
- **CI integration** via GitHub Actions for automated feature and training workflows.
- **MongoDB integration** for feature storage and model registry.

---

## 🏗️ System Architecture

### High‑Level Architecture

```text
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Open‑Meteo API │ ───▶ │  Feature Pipeline   │ ───▶ │   MongoDB (Features) │
│  (Air + Weather)│      │  (pipelines/)       │      │   + Model Registry   │
└─────────────────┘      └─────────────────────┘      └─────────────────────┘
                                                               │
                                                               ▼
┌─────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│  Streamlit      │ ───▶ │   FastAPI Service   │ ───▶ │  Trained ML Models  │
│  Dashboard      │      │   (api/)            │      │  (in MongoDB/local) │
└─────────────────┘      └─────────────────────┘      └─────────────────────┘

        ▲
        │
┌─────────────────┐
│ GitHub Actions  │
│  - Hourly       │
│    Feature Job  │
│  - Daily        |
|    Training Job │
│                 │
└─────────────────┘
```

### Components

- **Data Collection & Feature Engineering**
  - `pipelines/data_fetcher.py` – wraps Open‑Meteo APIs (air quality + weather).
  - `pipelines/data_cleaning.py` – data cleaning and deduplication.
  - `pipelines/aqi_calculator.py` – EPA‑style AQI computation when not provided.
  - `pipelines/feature_engineering.py` – engineered numerical and temporal features.
  - `pipelines/feature_pipeline.py` – orchestrator that:
    - fetches raw data,
    - cleans & calculates AQI,
    - engineers features,
    - stores them in MongoDB & local CSVs.

- **Feature Store & Model Registry**
  - `pipelines/mongodb_store.py` – handles:
    - inserting features with duplicate checking,
    - reading features for training,
    - persisting trained models and metadata,
    - retrieving the current “best” model.

- **Machine Learning Layer**
  - `pipelines/training_pipeline.py` – training orchestration:
    - loads features (from MongoDB or `data/features`),
    - prepares data with a time‑aware split,
    - trains multiple models,
    - builds an ensemble,
    - evaluates and saves models plus metrics.

- **API Layer (FastAPI)**
  - `api/main.py` – FastAPI app exposing:
    - `GET /health` – health and MongoDB status,
    - `GET /models` – list models and metrics from the registry,
    - `POST /predict` – N‑day AQI forecast based on current conditions.
  - Deployed on **Railway**, with `railway.toml` controlling startup and health checks.

- **Presentation Layer (Streamlit)**
  - `app/dashboard.py` – Streamlit dashboard that:
    - calls FastAPI for predictions and model metadata,
    - visualises AQI forecasts and model metrics,
    - shows alerts for unhealthy AQI levels.
  - `.streamlit/config.toml` – dashboard theme and server config.

- **Automation Layer (CI)**
  - `.github/workflows/feature_pipeline.yml` – hourly feature pipeline.
  - `.github/workflows/training_pipeline.yml` – training pipeline (on `main` push / manual).

---

## ✨ Key Features

### 1. Feature Pipeline

- **Combined Open‑Meteo ingestion**:
  - Air quality (`air_quality_url`),
  - Weather forecasts (`weather_url`),
  - Optional historical data (`historical_url`).
- **Cleaning & normalisation**:
  - Standard schema for pollutant and weather fields,
  - Duplicate removal and timestamp sorting,
  - Local backups in `data/raw/raw_data.csv`.
- **Feature engineering**:
  - AQI computation via `EPAAQICalculator` when needed.
  - Rich numeric feature set prepared for tabular models.
- **Feature storage**:
  - MongoDB Atlas (or local MongoDB) as the primary feature store.
  - Consolidated CSV backups in `data/features/`.

### 2. Machine Learning Models

- **Algorithms** (config‑driven via `config/config.yaml`):
  - Random Forest Regressor.
  - Ridge Regression with a full sklearn `Pipeline` (imputer + scaler + feature selection).
  - XGBoost Regressor with early stopping.
  - LightGBM Regressor (if installed and available).
  - Ensemble model (VotingRegressor) combining individual models.
- **Data preparation**:
  - Next‑day AQI target (\( \text{AQI}_{t+24h} \)) created from current AQI.
  - Non‑numeric columns removed.
  - Time‑aware train/test split based on timestamps.
- **Evaluation**:
  - Metrics (per model): RMSE, MAE, R² (and others as configured).
  - Consolidated metrics stored with the model in MongoDB.

### 3. Production API

- **FastAPI application** (`api/main.py`):
  - `GET /` – service info and endpoint index.
  - `GET /health` – health probe used by Railway.
  - `GET /models` – available models sorted by `trained_at`.
  - `POST /predict` – AQI forecast for the next N days.
- **Prediction logic**:
  - Fetches current combined data (air + weather) from Open‑Meteo.
  - Cleans and engineers features using the same code as training.
  - Loads the **best model** from the model registry (lowest valid RMSE).
  - Adjusts temporal features for each future day and returns AQI + category.
- **CORS**:
  - Allowed origins configured via `ALLOWED_ORIGINS` env (default: `*`).

### 4. Interactive Dashboard

- **Built with Streamlit** (`app/dashboard.py`):
  - Configuration sidebar showing city and coordinates from `config/config.yaml`.
  - Calls `GET /health`, `GET /models`, and `POST /predict` from the FastAPI API.
  - Metrics table for all tracked models (RMSE, MAE, R², MAPE).
  - Highlight of the best model based on RMSE.
  - Bar chart forecast for AQI by date with colour‑coded categories.
  - Alerts when predicted AQI exceeds “unhealthy” thresholds from config.

### 5. CI Integration

- **Hourly feature pipeline**:
  - `.github/workflows/feature_pipeline.yml`.
  - Uses a lightweight dependency set (no heavy DL frameworks).
  - Runs `python pipelines/feature_pipeline.py` with `MONGODB_URI` from secrets.
  - Uploads feature CSVs and logs as artifacts.

- **Daily Training pipeline**:
  - `.github/workflows/training_pipeline.yml`.
  - Creates a venv, installs focused ML dependencies (pandas, numpy, scikit‑learn, xgboost, lightgbm).
  - Tests MongoDB connectivity.
  - Runs the training pipeline (configurable to skip heavy models in CI).

---

## 🛠️ Technology Stack

### Core

- **Language**: Python 3.10+  
- **Configuration**: YAML (`config/config.yaml`) + `.env` via `python-dotenv`  

### Data & ML

- **Pandas**, **NumPy** – data handling and numerical features.  
- **scikit‑learn** – preprocessing, models, evaluation metrics.  
- **XGBoost**, **LightGBM** – gradient boosting models.  

### Storage

- **MongoDB Atlas / MongoDB** – feature store and model registry.  
- **PyMongo**, **Motor** (async driver) – database access.

### Backend

- **FastAPI** – REST API.  
- **Uvicorn** – ASGI server (used locally and by Railway).  

### Frontend

- **Streamlit** – dashboard UI.  
- **Plotly** – interactive charts.

### DevOps

- **GitHub Actions** – CI workflows.  
- **Railway** – FastAPI deployment (via `railway.toml`).  

---

## 🌐 Live Deployment

> Replace these placeholders with your actual deployment URLs.

- **Dashboard**: `https://rawalpindi-aqi-prediction.streamlit.app/`  
- **API Base URL (Railway)**: `https://aqi-prediction-production-5eee.up.railway.app/`

---

## 📦 Installation & Setup

### Prerequisites

- **Python** 3.10 or higher  
- **MongoDB Atlas** account (or a local MongoDB instance)  
- **Git**

### 1. Clone Repository

```bash
git clone <your-repo-url>
cd AQI-Prediction
```

### 2. Create and Activate Virtual Environment

```bash
python -m venv venv

# Windows (PowerShell)
venv\Scripts\Activate.ps1

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

#### Full project (dashboard + pipelines + local API)

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### API‑only (for Railway or separate API deployment)

```bash
cd api
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure Environment Variables

Create a `.env` file in the **project root** (next to `config/`):

```env
# Required: MongoDB connection
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/aqi_predictor?retryWrites=true&w=majority

# Optional: Open‑Meteo API key (not required for free tier)
OPENMETEO_API_KEY=your_key_if_any

# Optional: API URL used by Streamlit when running locally
FASTAPI_URL=http://localhost:8000

# Optional: CORS origins (comma‑separated list or *)
ALLOWED_ORIGINS=*
```

You can use `config/env.example` as a reference/template.

### 5. Verify Setup (Recommended)

Run the quick start script:

```bash
python scripts/quick_start.py
```

This will:

- Check that core Python dependencies are installed.  
- Verify presence of `config/config.yaml` and `.env`.  
- Create `data/features`, `data/raw`, `models`, and `logs` directories.  

### 6. Test MongoDB Connection

```bash
python scripts/test_mongodb.py
```

If credentials or network rules are misconfigured, this script gives detailed diagnostics and encoding guidance.

---

## 📁 Project Structure

```text
AQI-Prediction/
├── api/                       # FastAPI service
│   ├── main.py                # API endpoints and prediction logic
│   └── requirements.txt       # API‑specific dependencies
│
├── app/                       # Streamlit dashboard
│   ├── dashboard.py           # Main dashboard app
│   └── __init__.py
│
├── pipelines/                 # Core ML/data pipelines
│   ├── __init__.py
│   ├── data_fetcher.py        # Open‑Meteo API client
│   ├── data_cleaning.py       # Cleaning, deduplication utilities
│   ├── feature_engineering.py # Feature construction
│   ├── feature_pipeline.py    # Full feature pipeline orchestration
│   ├── aqi_calculator.py      # EPA AQI calculation
│   ├── mongodb_store.py       # MongoDB feature/model storage
│   ├── training_pipeline.py   # ML training and model registry
│   └── backfill.py            # Historical backfill (if used)
│
├── scripts/                   # Utility scripts
│   ├── quick_start.py         # Initial setup and checks
│   ├── setup_cloud.py         # Interactive cloud setup helper
│   ├── run_dashboard.py       # Convenience launcher for Streamlit
│   ├── test_mongodb.py        # MongoDB diagnostics and URI encoding
│   ├── run_optimized_backfill.py # Historical data backfill
│   ├── export_to_csv.py       # Export helpers
│   ├── check_data_quality.py  # Data quality checks
│   ├── check_overfitting.py   # Model overfitting inspection
│   └── clear_features.py      # Maintenance utilities
│
├── config/
│   ├── config.yaml            # Main configuration (APIs, Mongo, models, city)
│   ├── settings.py            # Loader for YAML + env overrides
│   └── env.example            # Example env file
│
├── .github/
│   └── workflows/
│       ├── feature_pipeline.yml   # Hourly feature pipeline
│       └── training_pipeline.yml  # Training job (push/manual)
│
├── .streamlit/
│   ├── config.toml            # Streamlit theme + server config
│   └── secrets.toml           # (for Streamlit Cloud, locally optional)
│
├── data/
│   ├── raw/                   # Raw data backups
│   └── features/              # Engineered features (CSV/parquet)
│
├── models/                    # Local model backups (if used)
├── railway.toml               # Railway deployment config for API
├── requirements.txt           # Project‑level dependencies
├── setup.py                   # (used in some environments)
└── .env                       # Local environment (not committed)
```

---

## 🚀 Usage Guide

### 1. Run Feature Pipeline

Generate latest features and store them:

```bash
python pipelines/feature_pipeline.py
```

This will:

1. Fetch combined air quality + weather data from Open‑Meteo.  
2. Clean and normalise the raw data, computing AQI if needed.  
3. Engineer a rich set of features for ML.  
4. Save raw data and features to `data/`.  
5. Insert new feature rows into MongoDB (with duplicate checks).

### 2. Train Models

Train all configured ML models and register the best one:

```bash
python pipelines/training_pipeline.py
```

This will:

1. Load features from MongoDB (or fallback to `data/features/`).  
2. Build a next‑day AQI prediction target.  
3. Split data into time‑aware train/test sets.  
4. Train multiple models (Random Forest, Ridge, XGBoost, LightGBM where available).  
5. Build an ensemble regressor.  
6. Evaluate models and select the best one by RMSE.  
7. Store the model, metrics, and feature list into MongoDB.

### 3. Run the API Locally

From the project root:

```bash
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

The API will be available at `http://localhost:8000`.

### 4. Run the Streamlit Dashboard Locally

With the API running:

```bash
# Option 1: use helper
python scripts/run_dashboard.py

# Option 2: direct command
streamlit run app/dashboard.py
```

Then open `http://localhost:8501` in your browser.

---

## 🤖 Machine Learning Models

### Algorithms

- **RandomForestRegressor**
- **Ridge Regression** (via sklearn `Pipeline` including imputation, scaling, and feature selection)
- **XGBRegressor** (XGBoost)
- **LightGBM Regressor** (if the package is installed)
- **VotingRegressor** ensemble over selected models

### Training Details

- **Target**: AQI 24 hours ahead (next‑day AQI) derived from current AQI.  
- **Split**: Time‑aware, based on the `timestamp` column to avoid leakage.  
- **Preprocessing**:
  - Numeric‑only features.
  - Median imputation and standard scaling (in Pipelines).
  - SelectKBest‑based feature selection for Ridge.
- **Model Selection**:
  - Best model chosen based on **lowest valid RMSE**.
  - Metrics and metadata stored in the model registry collection.

---

## 🔄 CI/CD Pipeline

### 1. Hourly Feature Pipeline

**File**: `.github/workflows/feature_pipeline.yml`  
**Trigger**: `cron: '0 * * * *'` (every hour) + manual dispatch.

Workflow summary:

- Checks out code and sets up Python 3.12.  
- Cleans disk space aggressively (for GitHub‑hosted runners).  
- Installs **only** the minimal dependencies needed for the feature pipeline (no `requirements.txt`).  
- Runs:

  ```bash
  python pipelines/feature_pipeline.py
  ```

  with `MONGODB_URI` from GitHub Secrets.  

- Uploads feature CSVs and logs as artifacts.  
- On failure, automatically opens a GitHub issue with a link to the failed run.

### 2. Training Pipeline

**File**: `.github/workflows/training_pipeline.yml`  
**Trigger**: daily + manual `workflow_dispatch`.

Workflow summary:

- Checks out the repository and sets up Python 3.11.  
- Creates a virtual environment and installs focused ML dependencies.  
- Tests MongoDB connectivity using `MongoDBStore`.  
- Runs the training pipeline (optionally with heavy models disabled in CI):

  ```python
  from pipelines.training_pipeline import TrainingPipeline

  pipeline = TrainingPipeline(train_lightgbm=False, train_xgboost=False)
  pipeline.run()
  ```

You can extend this job with a `schedule` block if you want a daily cron.

---

## 📡 API Documentation

### Base URL

```text
http://localhost:8000        # local
https://<your-railway-url>   # production
```

### 1. Root

**GET** `/`

Returns service metadata and endpoints.

### 2. Health Check

**GET** `/health`

Lightweight health check used by Railway.

**Sample Response**:

```json
{
  "status": "healthy",
  "mongodb": "connected",
  "timestamp": "2026-02-27T12:00:00.000000",
  "service": "AQI Prediction API"
}
```

### 3. List Models

**GET** `/models`

Returns list of available models, ordered by training time (newest first) and filtered to those with valid metrics.

**Sample Response**:

```json
{
  "models": [
    {
      "name": "random_forest",
      "trained_at": "2026-02-26T02:00:00",
      "metrics": {
        "rmse": 21.3,
        "mae": 17.2,
        "r2": 0.20,
        "mape": 18.6
      }
    }
  ]
}
```

### 4. Predict AQI

**POST** `/predict`

Forecast AQI for the next `forecast_days` days using the best available model.

**Request Body**:

```json
{
  "forecast_days": 3,
  "latitude": 33.5651,
  "longitude": 73.0169
}
```

If `latitude`/`longitude` are omitted, the system uses the default city from `config.yaml`.

**Sample Response**:

```json
{
  "predictions": [
    {
      "date": "2026-02-28",
      "day": 1,
      "predicted_aqi": 85.5,
      "category": "Moderate"
    }
  ],
  "current_aqi": 82.3,
  "model_name": "random_forest",
  "model_metrics": {
    "rmse": 21.3,
    "mae": 17.2,
    "r2": 0.20,
    "mape": 18.6
  },
  "generated_at": "2026-02-27T12:00:00.000000"
}
```

---

