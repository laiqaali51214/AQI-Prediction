# AQI Prediction – Step-by-Step Run Guide

This guide walks you through understanding and running the project locally from scratch.

---

## 1. What the Project Does

**10pearlsAQI** predicts Air Quality Index (AQI) for a city (e.g. Rawalpindi/Karachi) for the next 1–7 days.

- **Data**: Open-Meteo API (air quality + weather).
- **Features**: 69 engineered features (time, pollutants, weather, rolling stats, etc.).
- **Models**: Random Forest, Ridge, XGBoost, LightGBM, plus an Ensemble.
- **Storage**: MongoDB Atlas (features + model registry); local CSV backup.
- **API**: FastAPI for predictions.
- **UI**: Streamlit dashboard.

**Rough flow**: Fetch data → Clean → Engineer features → Store (MongoDB + CSV) → Train models → Serve via API → Dashboard calls API.

---

## 2. Prerequisites

- **Python 3.10+** (3.12 recommended)
- **MongoDB Atlas** account (free tier): [mongodb.com/cloud/atlas](https://www.mongodb.com/cloud/atlas)
- **Git** (already cloned)

---

## 3. Step-by-Step Setup and Run

All commands below are from the **project root** (`AQI-Prediction`).

### Step 1: Open terminal at project root

```powershell
cd "c:\Users\Laiqa Ali\OneDrive - National University of Sciences & Technology\10PearlsPakistan-DataScience\AQI-Prediction"
```

### Step 2: Create and activate virtual environment

```powershell
python -m venv venv
.\venv\Scripts\activate
```

### Step 3: Install dependencies

```powershell
pip install -r requirements.txt
```

### Step 4: Configure environment

1. Copy the example env file:
   ```powershell
   copy config\env.example .env
   ```
2. Edit `.env` and set:
   - **MONGODB_URI** (required): your MongoDB Atlas connection string.
   - Optionally: `FASTAPI_URL=http://localhost:8000` for the dashboard.

   Example:
   ```env
   MONGODB_URI=mongodb+srv://USER:PASSWORD@cluster.mongodb.net/aqi_predictor?retryWrites=true&w=majority
   FASTAPI_URL=http://localhost:8000
   ```

   If your password has special characters (`@`, `#`, etc.), encode them (e.g. `@` → `%40`) or run:
   ```powershell
   python scripts/test_mongodb.py encode "YOUR_FULL_URI"
   ```
   and use the printed URI in `.env`.

### Step 5: Configure city (optional)

Edit `config/config.yaml` if you want to change city:

```yaml
city:
  name: "Rawalpindi"
  latitude: 33.5651
  longitude: 73.0169
  timezone: "Asia/Karachi"
```

### Step 6: Verify setup

```powershell
python scripts/quick_start.py
```

This checks dependencies, config, and creates `data/features`, `data/raw`, `models`, `logs`.

### Step 7: Test MongoDB connection

```powershell
python scripts/test_mongodb.py
```

If it fails, run diagnostics:

```powershell
python scripts/test_mongodb.py diagnose
```

Fix Atlas Network Access (whitelist IP or `0.0.0.0/0`) and credentials as needed.

### Step 8: Get some data and features

**Option A – Current data only (quick)**  
Runs the feature pipeline once (current weather + AQI → features → MongoDB + CSV):

```powershell
python pipelines/feature_pipeline.py
```

**Option B – Historical backfill (for better models)**  
Fetches up to a year of history, then you run the feature pipeline (or backfill script may integrate it). Recommended before first training:

```powershell
python scripts/run_optimized_backfill.py --resume
```

Then run the feature pipeline so that backfilled raw data is turned into features and stored. (If the backfill script already writes features, you can skip this once.)

### Step 9: Train models

Uses features from MongoDB (or local CSV fallback), trains all algorithms, picks best by RMSE, and stores in MongoDB:

```powershell
python pipelines/training_pipeline.py
```

### Step 10: Start the API

In a **first** terminal (from project root):

```powershell
.\venv\Scripts\activate
cd api
uvicorn main:app --host 0.0.0.0 --port 8000
```

API base: **http://localhost:8000**  
Docs: **http://localhost:8000/docs**

### Step 11: Start the dashboard

In a **second** terminal (from project root):

```powershell
.\venv\Scripts\activate
streamlit run app/dashboard.py
```

Or:

```powershell
python scripts/run_dashboard.py
```

Open **http://localhost:8501**, choose forecast days, and click “Get Predictions”.

---

## 4. Quick Reference – What to Run When

| Goal                         | Command                                      |
|-----------------------------|----------------------------------------------|
| Check setup                 | `python scripts/quick_start.py`              |
| Test MongoDB                | `python scripts/test_mongodb.py`             |
| Fetch current + features    | `python pipelines/feature_pipeline.py`       |
| Backfill history            | `python scripts/run_optimized_backfill.py --resume` |
| Train / retrain models       | `python pipelines/training_pipeline.py`      |
| Start API                   | `cd api && uvicorn main:app --host 0.0.0.0 --port 8000` |
| Start dashboard             | `streamlit run app/dashboard.py`              |

---

## 5. Project Layout (relevant parts)

```
AQI-Prediction/
├── config/
│   ├── config.yaml      # City, APIs, models, pipeline settings
│   └── env.example      # Template for .env
├── pipelines/
│   ├── data_fetcher.py       # Open-Meteo fetch
│   ├── data_cleaning.py     # Clean raw data
│   ├── feature_engineering.py # 69 features
│   ├── feature_pipeline.py  # Fetch → clean → features → store
│   ├── training_pipeline.py # Load features → train → store best model
│   ├── mongodb_store.py     # MongoDB read/write
│   ├── aqi_calculator.py    # EPA AQI
│   └── backfill.py          # Historical data backfill
├── api/
│   └── main.py          # FastAPI app (predict, health, models)
├── app/
│   └── dashboard.py     # Streamlit UI
├── scripts/
│   ├── quick_start.py   # Setup check
│   ├── test_mongodb.py  # MongoDB test/diagnose/encode
│   ├── run_optimized_backfill.py
│   └── run_dashboard.py
├── data/raw/            # Raw CSV backup
├── data/features/       # Features CSV backup
├── models/              # Local model artifacts
├── .env                 # Your secrets (create from config/env.example)
└── requirements.txt
```

---

## 6. Troubleshooting

- **MongoDB connection failed**  
  Use `python scripts/test_mongodb.py diagnose`. Check URI, encoding, Atlas IP whitelist, and user permissions.

- **No features / training fails**  
  Ensure you’ve run the feature pipeline (and backfill if you want more data) so MongoDB or `data/features/` has data.

- **“No trained model found” in API**  
  Run `python pipelines/training_pipeline.py` and ensure it finishes without errors.

- **Dashboard can’t reach API**  
  Start the API first; in `.env` set `FASTAPI_URL=http://localhost:8000`.

- **Import errors**  
  Always activate the venv and run commands from the project root so `config` and `pipelines` resolve correctly.

---

## 7. CI/CD (reference)

- **Feature pipeline**: runs hourly (e.g. GitHub Actions).
- **Training pipeline**: runs daily.
See `.github/workflows/` and README for details.
