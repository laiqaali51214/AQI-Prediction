# Complete Setup Guide

This comprehensive guide covers all setup steps for the AQI Predictor project.

## Table of Contents

1. [Prerequisites](#prerequisites)
2. [API Keys Setup](#api-keys-setup)
3. [MongoDB Setup](#mongodb-setup)
4. [Environment Configuration](#environment-configuration)
5. [City Configuration](#city-configuration)
6. [GitHub Actions Setup](#github-actions-setup)
7. [VS Code MongoDB Extension (Optional)](#vs-code-mongodb-extension-optional)
8. [Verification](#verification)

---

## Prerequisites

### Required Accounts

- GitHub account (for repository and CI/CD)
- MongoDB Atlas account (for database)
- Open-Meteo API (no account needed for free tier)

### Required Software

- Python 3.12 or higher
- Git
- Code editor (VS Code recommended)

### Verify Python Installation

```bash
python --version
```

Should show Python 3.12 or higher.

---

## API Keys Setup

### Open-Meteo API (Optional)

**Status**: No API key required for non-commercial use.

Open-Meteo API is **free for non-commercial use** and does not require an API key. You can start using it immediately.

**Free Tier Limits:**
- Up to 10,000 requests per day
- Suitable for development and personal projects
- No registration required

**When You Need an API Key:**
- Commercial applications
- High-volume requests (>10,000/day)
- Dedicated servers or premium features

**To get an API key (if needed):**
1. Visit: https://open-meteo.com/en/pricing
2. Choose a plan (Standard, Professional, etc.)
3. Register and get your API key
4. Add it to your `.env` file as `OPENMETEO_API_KEY`

**API Endpoints Used:**
- Air Quality API: `https://air-quality-api.open-meteo.com/v1/air-quality`
- Weather Forecast API: `https://api.open-meteo.com/v1/forecast`
- Historical Weather Archive: `https://archive-api.open-meteo.com/v1/archive`

**Configuration:**
The API endpoints are configured in `config/config.yaml`. The data fetcher automatically uses the API key if provided in the environment variable. If no key is provided, it works without authentication (free tier).

---

## MongoDB Setup

### Step 1: Create MongoDB Atlas Account

1. Visit: https://www.mongodb.com/cloud/atlas
2. Click "Try Free" or "Sign Up"
3. Complete the registration form
4. Verify your email address

### Step 2: Create MongoDB Cluster

1. After login, click "Build a Database"
2. Choose "M0 Free" tier (suitable for development)
3. Select your preferred cloud provider (AWS, GCP, or Azure)
4. Choose a region closest to your deployment location
5. Name your cluster (e.g., "AQI-Predictor-Cluster")
6. Click "Create Cluster"
7. Wait for cluster creation (takes 3-5 minutes)

### Step 3: Configure Database Access

1. In MongoDB Atlas dashboard, go to "Database Access"
2. Click "Add New Database User"
3. Choose "Password" authentication
4. Enter a username (e.g., "aqi_user")
5. Generate a secure password or create your own
6. Set user privileges to "Read and write to any database"
7. Click "Add User"
8. **Save the username and password securely**

### Step 4: Configure Network Access

1. Go to "Network Access" in MongoDB Atlas
2. Click "Add IP Address"
3. For development, click "Allow Access from Anywhere" (0.0.0.0/0)
4. For production, add specific IP addresses
5. Click "Confirm"

### Step 5: Get Connection String

1. Go to "Database" in MongoDB Atlas
2. Click "Connect" on your cluster
3. Choose "Connect your application"
4. Select "Python" as driver
5. Copy the connection string
6. Replace `<password>` with your database user password
7. Replace `<dbname>` with your database name (e.g., "aqi_predictor")

**Example connection string:**
```
mongodb+srv://aqi_user:your_password@cluster0.xxxxx.mongodb.net/aqi_predictor?retryWrites=true&w=majority
```

**Important**: Special characters in the password must be URL-encoded:
- `@` becomes `%40`
- `#` becomes `%23`
- `$` becomes `%24`

You can use `python scripts/test_mongodb.py encode <connection_string>` to automatically encode your connection string.

---

## Environment Configuration

### Step 1: Create .env File

Create a `.env` file in the project root:

```bash
# Copy the example file
cp config/env.example .env
```

### Step 2: Add Environment Variables

Edit `.env` and add your configuration:

```env
# MongoDB Configuration (Required)
MONGODB_URI=mongodb+srv://username:password@cluster.mongodb.net/aqi_predictor?retryWrites=true&w=majority

# Open-Meteo API Key (Optional - only for commercial use)
OPENMETEO_API_KEY=

# FastAPI Configuration (for Streamlit)
FASTAPI_URL=http://localhost:8000
```

**Important**: Never commit the `.env` file to version control. It should already be in `.gitignore`.

### Step 3: Test MongoDB Connection

```bash
python scripts/test_mongodb.py
```

**If connection fails:**
- Run diagnostics: `python scripts/test_mongodb.py diagnose`
- Check IP whitelist in MongoDB Atlas
- Verify connection string encoding
- Check username and password

---

## City Configuration

### Step 1: Find Your City Coordinates

**Option 1**: Use online tool
- Visit: https://www.latlong.net/
- Search for your city

**Option 2**: Use Google Maps
- Right-click on your city
- Click on coordinates

### Step 2: Find Your Timezone

Common timezones:
- `Asia/Karachi` (Pakistan)
- `America/New_York` (Eastern US)
- `America/Chicago` (Central US)
- `America/Los_Angeles` (Pacific US)
- `Europe/London` (UK)
- `Asia/Kolkata` (India)

Full list: https://en.wikipedia.org/wiki/List_of_tz_database_time_zones

### Step 3: Update Configuration

Edit `config/config.yaml`:

```yaml
city:
  name: "Your City Name"
  latitude: 40.7128
  longitude: -74.0060
  timezone: "America/New_York"
```

---

## GitHub Actions Setup

### Step 1: Add Repository Secrets

1. Go to your GitHub repository
2. Navigate to **Settings** → **Secrets and variables** → **Actions**
3. Click **New repository secret** and add the following secrets:

#### Required Secrets

- **`MONGODB_URI`**: Your MongoDB Atlas connection string
  - Format: `mongodb+srv://username:password@cluster.mongodb.net/database_name?retryWrites=True&w=majority`
  - Note: Special characters in the password must be URL-encoded (@ = %40, # = %23)

- **`OPENMETEO_API_KEY`** (Optional): Open-Meteo API key for commercial use
  - Only required if you exceed the free tier limits (>10,000 requests/day)
  - Can be left empty for non-commercial use

### Step 2: Verify Workflow Files

The workflow files are already configured in `.github/workflows/`:

- **Feature Pipeline**: Runs every hour at minute 0 (`0 * * * *`)
- **Training Pipeline**: Runs daily at 2 AM UTC (`0 2 * * *`)

### Step 3: Enable GitHub Actions

1. Go to **Actions** tab in your GitHub repository
2. If workflows are disabled, click **"I understand my workflows, go ahead and enable them"**
3. Workflows will run automatically based on their schedules

### Step 4: Manual Triggering

Both workflows can be manually triggered:

1. Go to **Actions** tab in your GitHub repository
2. Select the workflow you want to run (Feature Pipeline or Training Pipeline)
3. Click **Run workflow** → **Run workflow**

### Workflow Details

#### Feature Pipeline Workflow

- **Schedule**: Every hour
- **Timeout**: 30 minutes
- **Steps**:
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies
  4. Create data directories
  5. Run feature pipeline
  6. Upload artifacts (CSV files and logs)
  7. Create GitHub issue on failure

- **Artifacts**: 
  - Feature CSV files from `data/features/`
  - Log files from `logs/`
  - Retention: 7 days

#### Training Pipeline Workflow

- **Schedule**: Daily at 2 AM UTC
- **Timeout**: 60 minutes
- **Steps**:
  1. Checkout code
  2. Set up Python 3.12
  3. Install dependencies
  4. Create model directories
  5. Run training pipeline
  6. Upload artifacts (trained models and logs)
  7. Create GitHub issue on failure

- **Artifacts**:
  - Trained model files (`.pkl`, `.json`) from `models/`
  - Log files from `logs/`
  - Retention: 30 days

### Monitoring Workflows

#### View Workflow Runs

1. Go to the **Actions** tab in your GitHub repository
2. You'll see a list of all workflow runs with their status:
   - ✅ Green checkmark = Success
   - ❌ Red X = Failure
   - 🟡 Yellow circle = In progress

#### View Workflow Logs

1. Click on a workflow run to see details
2. Click on a job to see individual steps
3. Click on a step to see detailed logs

#### Failure Notifications

When a workflow fails, a GitHub issue is automatically created with:
- Title: `[Pipeline Name] Failed - [Timestamp]`
- Body: Link to the failed workflow run

### Customizing Schedules

To change the schedule, edit the `cron` expression in the workflow files:

#### Cron Syntax

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday to Saturday)
│ │ │ │ │
* * * * *
```

#### Examples

- `0 * * * *` - Every hour at minute 0
- `0 2 * * *` - Daily at 2 AM UTC
- `0 */6 * * *` - Every 6 hours
- `0 0 * * 1` - Every Monday at midnight

---

## VS Code MongoDB Extension (Optional)

This guide explains how to connect to MongoDB Atlas using the MongoDB extension for Visual Studio Code.

### Step 1: Install MongoDB Extension

1. Open Visual Studio Code
2. Go to Extensions (Ctrl+Shift+X or Cmd+Shift+X)
3. Search for "MongoDB for VS Code" by MongoDB
4. Click "Install"
5. Wait for installation to complete

### Step 2: Connect to MongoDB Atlas

#### Option A: Using Connection String (Recommended)

1. In VS Code, open the Command Palette (Ctrl+Shift+P or Cmd+Shift+P)
2. Type "MongoDB: Add Connection"
3. Select "MongoDB: Add Connection"
4. Choose "Connect with connection string"
5. Paste your MongoDB Atlas connection string:
   ```
   mongodb+srv://username:password@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority
   ```
   **Note**: Replace `cluster0.xxxxx.mongodb.net` with your actual cluster address
   **Important**: Password must be URL-encoded:
   - `@` becomes `%40`
   - `#` becomes `%23`

6. Enter a connection name (e.g., "AQI Predictor - Atlas")
7. Click "Connect"

#### Option B: Using Connection Form

1. Open Command Palette (Ctrl+Shift+P)
2. Type "MongoDB: Add Connection"
3. Select "MongoDB: Add Connection"
4. Choose "Fill in connection details individually"
5. Enter the following:
   - **Connection Name**: AQI Predictor - Atlas
   - **Connection String**: `mongodb+srv://`
   - **Host**: `cluster0.xxxxx.mongodb.net` (your cluster address)
   - **Port**: Leave empty (defaults to 27017)
   - **Authentication**: Username / Password
   - **Username**: Your database username
   - **Password**: Your database password (use actual password, not URL-encoded)
   - **Authentication Database**: `admin` (or leave default)
   - **Replica Set**: Leave empty
   - **SSL**: Enabled (required for Atlas)
   - **Read Preference**: Primary

6. Click "Connect"

### Step 3: Verify Connection

After connecting, you should see:

1. MongoDB icon in the left sidebar
2. Your connection listed under "MongoDB Connections"
3. Expand the connection to see:
   - Databases
   - Collections
   - Documents

### Step 4: Browse Your Database

1. Expand your connection in the MongoDB sidebar
2. Navigate to your database (e.g., `aqi_predictor`)
3. View collections:
   - `aqi_features` - Feature data
   - `aqi_models` - Trained models
   - `pipeline_metadata` - Pipeline execution metadata

### Step 5: View and Edit Documents

1. Click on a collection to view documents
2. Right-click on a document to:
   - View document
   - Edit document
   - Delete document
   - Copy document

### Troubleshooting VS Code Extension

#### Connection Fails

**Problem**: Cannot connect to MongoDB Atlas
- **Solution**: 
  - Verify IP address is added to Network Access (0.0.0.0/0 for development)
  - Check username and password are correct
  - Ensure connection string is properly formatted
  - Verify cluster is running (not paused)

#### Authentication Error

**Problem**: Authentication failed
- **Solution**:
  - Verify username and password
  - Check database user has correct permissions
  - Ensure password special characters are URL-encoded in connection string

#### SSL/TLS Error

**Problem**: SSL connection error
- **Solution**:
  - Ensure SSL is enabled in connection settings
  - MongoDB Atlas requires SSL/TLS connections
  - Check firewall settings

#### Extension Not Working

**Problem**: MongoDB extension not appearing
- **Solution**:
  - Reload VS Code window (Ctrl+Shift+P → "Developer: Reload Window")
  - Reinstall the extension
  - Check VS Code version compatibility

### Useful Extension Features

1. **MongoDB Playground**: Write and test MongoDB queries
2. **Document Explorer**: Browse collections and documents visually
3. **Query History**: View previously executed queries
4. **Schema Analysis**: Analyze collection schemas
5. **Export/Import**: Export collections to JSON or import data

---

## Verification

### Step 1: Test MongoDB Connection

```bash
python scripts/test_mongodb.py
```

Expected output: Connection successful message

### Step 2: Test Feature Pipeline

```bash
python pipelines/feature_pipeline.py
```

This should:
- Fetch data from APIs
- Generate features
- Save to MongoDB and CSV

### Step 3: Test Training Pipeline

```bash
python pipelines/training_pipeline.py
```

This will:
- Load features from MongoDB
- Train ML models
- Save the best model

### Step 4: Test FastAPI Service Locally

```bash
python api/main.py
```

Or using uvicorn:
```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

Test the API:
```bash
curl http://localhost:8000/health
```

### Step 5: Test Streamlit Dashboard Locally

```bash
streamlit run app/dashboard.py
```

Open http://localhost:8501 in your browser.

---

## Troubleshooting

### MongoDB Connection Issues

- Verify connection string format
- Check network access settings in MongoDB Atlas
- Ensure username and password are correct
- Verify database name matches configuration
- Run diagnostics: `python scripts/test_mongodb.py diagnose`

### API Connection Issues

- Verify API keys are correct (if using commercial tier)
- Check API rate limits
- Verify network connectivity
- Check API service status

### GitHub Actions Issues

- Verify secrets are added correctly
- Check workflow file syntax
- Verify Python version in workflow
- Check workflow logs for errors
- Ensure GitHub Actions is enabled in repository settings

### Workflow Not Running

1. **Check schedule**: Scheduled workflows only run if the repository has been active in the last 60 days
2. **Check secrets**: Ensure all required secrets are configured
3. **Check permissions**: Ensure GitHub Actions is enabled in repository settings

### Workflow Failing

1. **Check logs**: Review the workflow logs in the Actions tab
2. **Common issues**:
   - Missing or incorrect `MONGODB_URI` secret
   - Network timeouts (increase timeout in workflow file)
   - Missing dependencies (check `requirements.txt`)
   - Data directory permissions

---

## Security Best Practices

1. Never commit `.env` files to version control
2. Use strong passwords for database users
3. Restrict MongoDB network access in production (avoid 0.0.0.0/0)
4. Use environment variables for all secrets
5. Regularly rotate API keys and database credentials
6. Monitor API usage for anomalies
7. Use HTTPS for all API endpoints
8. Implement rate limiting for production APIs
9. Review workflow permissions and limit access as needed

---

## Next Steps

After setup is complete:

1. **Run feature pipeline** to start collecting data
2. **Backfill historical data** for training (see EXECUTION_ORDER.md)
3. **Train models** on the collected data
4. **Launch dashboard** to see predictions
5. **Monitor GitHub Actions** for automated runs

For execution order and detailed workflow, see [EXECUTION_ORDER.md](EXECUTION_ORDER.md).
