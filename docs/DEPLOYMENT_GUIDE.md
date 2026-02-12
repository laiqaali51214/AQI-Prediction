# Deployment Guide

This guide covers deploying both the FastAPI backend and Streamlit dashboard separately to cloud platforms.

## Overview

- **FastAPI Backend**: Deployed to Railway, Render, or Heroku
- **Streamlit Dashboard**: Deployed to Streamlit Cloud
- **MongoDB**: Already configured on MongoDB Atlas

---

## Part 1: Deploy FastAPI Backend

### Option A: Railway (Recommended)

#### Step 1: Create Railway Account
1. Go to [railway.app](https://railway.app)
2. Sign in with GitHub
3. Click "New Project"

#### Step 2: Deploy from GitHub
1. Select "Deploy from GitHub repo"
2. Choose your repository (`10pearlsAQI`)
3. Railway will detect the `railway.json` configuration

#### Step 3: Configure Environment Variables
1. Go to your service → **Variables** tab
2. Add the following environment variables:
   ```
   MONGODB_URI=your_mongodb_connection_string
   ALLOWED_ORIGINS=*
   PORT=8000
   PYTHONPATH=/app
   ```

#### Step 4: Deploy
1. Railway will automatically build and deploy
2. Wait for deployment to complete
3. Copy the generated URL (e.g., `https://your-api.railway.app`)

#### Step 5: Test API
```bash
# Health check
curl https://your-api.railway.app/health

# Test prediction
curl -X POST https://your-api.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3}'
```

---

### Option B: Render

#### Step 1: Create Render Account
1. Go to [render.com](https://render.com)
2. Sign in with GitHub

#### Step 2: Create Web Service
1. Click "New +" → "Web Service"
2. Connect your GitHub repository
3. Select your repository

#### Step 3: Configure Service
- **Name**: `aqi-prediction-api`
- **Environment**: `Python 3`
- **Build Command**: `pip install -r api/requirements.txt`
- **Start Command**: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`

#### Step 4: Set Environment Variables
Go to **Environment** tab and add:
```
MONGODB_URI=your_mongodb_connection_string
ALLOWED_ORIGINS=*
PYTHONPATH=/opt/render/project/src
```

#### Step 5: Deploy
1. Click "Create Web Service"
2. Wait for deployment
3. Copy the URL (e.g., `https://aqi-prediction-api.onrender.com`)

---

### Option C: Heroku

#### Step 1: Install Heroku CLI
```bash
# Download from https://devcenter.heroku.com/articles/heroku-cli
```

#### Step 2: Login to Heroku
```bash
heroku login
```

#### Step 3: Create Heroku App
```bash
cd /path/to/10pearlsAQI
heroku create your-api-name
```

#### Step 4: Set Environment Variables
```bash
heroku config:set MONGODB_URI=your_mongodb_connection_string
heroku config:set ALLOWED_ORIGINS=*
```

#### Step 5: Deploy
```bash
git push heroku main
```

#### Step 6: Check Logs
```bash
heroku logs --tail
```

---

## Part 2: Deploy Streamlit Dashboard

### Step 1: Prepare for Streamlit Cloud

1. **Ensure code is pushed to GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Verify `.streamlit/config.toml` exists** (already created)

### Step 2: Deploy to Streamlit Cloud

1. **Go to Streamlit Cloud**
   - Visit [streamlit.io/cloud](https://streamlit.io/cloud)
   - Sign in with GitHub

2. **Create New App**
   - Click "New app"
   - Select your repository: `SagarChhabriya/10pearlsAQI`
   - Set **Main file path**: `app/dashboard.py`
   - Set **App URL** (optional): `your-app-name`

3. **Configure Secrets (Recommended)**
   - Click "Advanced settings" or the "⋮" menu → "Settings"
   - Go to the "Secrets" tab
   - Add secrets in TOML format:
     ```toml
     FASTAPI_URL = "https://your-api.railway.app"
     MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
     ```
   - **Important**: 
     - Replace `https://your-api.railway.app` with your actual FastAPI URL from Part 1
     - Replace `MONGODB_URI` with your actual MongoDB connection string
     - Secrets are encrypted and secure (better than environment variables)
   
   **Alternative: Environment Variables**
   - If you prefer environment variables, you can add them in the "Environment variables" section instead
   - But Streamlit secrets are recommended for better security

4. **Deploy**
   - Click "Deploy!"
   - Wait for deployment (usually 2-3 minutes)
   - Your dashboard will be available at: `https://your-app-name.streamlit.app`

---

## Part 3: Configure CORS (If Needed)

If you want to restrict CORS to specific origins:

### For FastAPI (Railway/Render/Heroku):
Set environment variable:
```
ALLOWED_ORIGINS=https://your-dashboard.streamlit.app,https://another-domain.com
```

### For Streamlit:
No CORS configuration needed (it's a client making requests)

---

## Part 4: Verify Deployment

### 1. Test FastAPI Backend

```bash
# Health check
curl https://your-api.railway.app/health

# List models
curl https://your-api.railway.app/models

# Get predictions
curl -X POST https://your-api.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3}'
```

### 2. Test Streamlit Dashboard

1. Open your Streamlit Cloud URL
2. Check "API Status" in the sidebar (should show "API Connected")
3. Click "Get Predictions" button
4. Verify predictions are displayed

---

## Troubleshooting

### FastAPI Issues

**Problem**: API returns 500 errors
- **Solution**: Check logs on your platform (Railway/Render/Heroku)
- Verify `MONGODB_URI` is set correctly
- Ensure MongoDB Atlas allows connections from cloud IPs (0.0.0.0/0)

**Problem**: CORS errors
- **Solution**: Set `ALLOWED_ORIGINS` environment variable to include your Streamlit URL

**Problem**: Models not loading
- **Solution**: Verify MongoDB connection and that models exist in the database

### Streamlit Issues

**Problem**: "API Unavailable" error
- **Solution**: 
  - Verify `FASTAPI_URL` environment variable is set correctly
  - Check that FastAPI service is running
  - Test FastAPI health endpoint manually

**Problem**: Predictions not loading
- **Solution**:
  - Check browser console for errors
  - Verify API URL is accessible
  - Check Streamlit Cloud logs

**Problem**: MongoDB connection errors
- **Solution**: 
  - Verify `MONGODB_URI` is set in Streamlit Cloud environment variables
  - Ensure MongoDB Atlas network access allows Streamlit Cloud IPs

---

## Environment Variables Summary

### FastAPI Backend
```
MONGODB_URI=your_mongodb_connection_string
ALLOWED_ORIGINS=* (or specific URLs)
PORT=8000 (usually set automatically)
PYTHONPATH=/app (or platform-specific path)
```

### Streamlit Dashboard (Using Secrets - Recommended)
```toml
# In Streamlit Cloud: Settings → Secrets tab
FASTAPI_URL = "https://your-api.railway.app"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
```

**Alternative: Environment Variables**
```
MONGODB_URI=your_mongodb_connection_string
FASTAPI_URL=https://your-api.railway.app
```

---

## Cost Estimates

### Free Tiers Available:
- **Railway**: $5/month free credit (usually enough for small apps)
- **Render**: Free tier available (with limitations)
- **Streamlit Cloud**: Free tier available
- **Heroku**: No free tier (paid only)

### Recommended for Free:
- **FastAPI**: Railway or Render free tier
- **Streamlit**: Streamlit Cloud free tier

---

## Next Steps After Deployment

1. **Monitor Logs**: Regularly check logs for errors
2. **Set Up Alerts**: Configure email notifications for failures
3. **Update Documentation**: Update README with live URLs
4. **Test Regularly**: Verify both services are working
5. **Backup**: Ensure MongoDB backups are configured

---

## Quick Reference

### FastAPI URLs
- Railway: `https://your-api.railway.app`
- Render: `https://your-api.onrender.com`
- Heroku: `https://your-api.herokuapp.com`

### Streamlit URL
- Streamlit Cloud: `https://your-app.streamlit.app`

### Test Commands
```bash
# Health check
curl https://your-api-url/health

# Get predictions
curl -X POST https://your-api-url/predict \
  -H "Content-Type: application/json" \
  -d '{"forecast_days": 3}'
```

---

## Support

If you encounter issues:
1. Check platform-specific logs
2. Verify environment variables are set
3. Test API endpoints manually
4. Check MongoDB connection
5. Review error messages in browser console (for Streamlit)
