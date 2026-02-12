# Documentation

This directory contains all project documentation for the AQI Predictor system.

## Quick Start

1. **Setup**: See [SETUP_GUIDE.md](SETUP_GUIDE.md) for complete setup instructions
2. **Execution**: See [EXECUTION_ORDER.md](EXECUTION_ORDER.md) for step-by-step execution order
3. **Technical Details**: See [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) for technical documentation

## Documentation Files

### [SETUP_GUIDE.md](SETUP_GUIDE.md)
Complete setup guide covering:
- API keys setup (Open-Meteo)
- MongoDB Atlas configuration
- Environment variables
- City configuration
- GitHub Actions setup
- VS Code MongoDB extension (optional)
- Verification and troubleshooting

### [EXECUTION_ORDER.md](EXECUTION_ORDER.md)
Step-by-step execution order guide covering:
- Initial setup and configuration
- Data collection (historical backfill)
- Model training
- API service startup
- Dashboard launch
- Automated pipelines (CI/CD)
- Complete execution checklist

### [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md)
Technical documentation covering:
- Dataset description and structure
- Preprocessing and EDA summary
- Model performance improvements
- Project structure and architecture

### [COMMIT_GUIDE.md](COMMIT_GUIDE.md)
Git commit order guide covering:
- Recommended commit order (8 phases)
- Files to commit vs. files to exclude
- Quick commit script
- Best practices and verification steps

### [WORKFLOW_MONITORING.md](WORKFLOW_MONITORING.md)
GitHub Actions workflow monitoring guide covering:
- How to check if workflows are executing on schedule
- Viewing workflow runs and logs
- Verifying scheduled runs
- Troubleshooting common issues
- Timezone conversion for cron schedules

### [SUPERVISOR_REPORT.md](SUPERVISOR_REPORT.md)
Comprehensive project report for supervisors covering:
- Executive summary and key achievements
- System architecture and components
- Data pipeline and feature engineering
- Machine learning models and performance
- Deployment and operations
- Challenges and solutions
- Future enhancements

## Quick Reference

### Essential Commands

```bash
# Test MongoDB connection
python scripts/test_mongodb.py

# Backfill historical data
python scripts/run_optimized_backfill.py --resume

# Run feature pipeline
python pipelines/feature_pipeline.py

# Train models
python pipelines/training_pipeline.py

# Start API service
python api/main.py

# Start dashboard
streamlit run app/dashboard.py
```

### Configuration Files

- `.env` - Environment variables (MongoDB URI, API keys)
- `config/config.yaml` - Project configuration (city, models, pipelines)

### Key Directories

- `pipelines/` - Reusable core modules (data fetching, cleaning, feature engineering, training)
- `scripts/` - Standalone utility scripts (testing, maintenance, execution)
- `api/` - FastAPI prediction service
- `app/` - Streamlit dashboard
- `data/` - Data storage (raw data, features)
- `models/` - Trained model storage

## Recommended Reading Order

1. **Start Here**: [SETUP_GUIDE.md](SETUP_GUIDE.md) - Complete setup instructions
2. **Then**: [EXECUTION_ORDER.md](EXECUTION_ORDER.md) - How to run the project
3. **Reference**: [TECHNICAL_DOCS.md](TECHNICAL_DOCS.md) - Technical details when needed

## Project Overview

The AQI Predictor is an automated air quality forecasting system that:

1. **Collects Data**: Fetches air quality and weather data from Open-Meteo API
2. **Processes Data**: Cleans data and engineers features for machine learning
3. **Trains Models**: Trains multiple ML models (Random Forest, Ridge, XGBoost, LightGBM, Ensemble)
4. **Makes Predictions**: Provides AQI forecasts for the next 1-7 days
5. **Displays Results**: Interactive dashboard for visualization

### Key Features

- **Automated Pipelines**: GitHub Actions run feature pipeline hourly and training pipeline daily
- **MongoDB Storage**: Features and models stored in MongoDB Atlas
- **Local Backup**: CSV files maintained for local systems
- **Real-time Predictions**: FastAPI service for on-demand predictions
- **Interactive Dashboard**: Streamlit web interface

## Support

For issues or questions:
- Check the relevant documentation file
- Review error logs
- Check GitHub Actions logs for CI/CD issues
- Verify configuration files are correct
