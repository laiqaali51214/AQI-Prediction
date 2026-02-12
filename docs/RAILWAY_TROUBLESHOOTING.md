# Railway Deployment Troubleshooting Guide

## Common Issues and Solutions

### 502 Bad Gateway Error

**What it means:**
- Railway's proxy can't reach your FastAPI backend
- The service might be crashing on startup
- The service might be taking too long to start
- There might be a configuration issue

**How to diagnose:**

1. **Check Railway Logs**
   - Go to your Railway project dashboard
   - Click on your service
   - Go to the "Logs" tab
   - Look for error messages, especially during startup

2. **Check Service Status**
   - In Railway dashboard, check if the service shows as "Running" or "Crashed"
   - If it's crashed, check the logs for the error

3. **Test Health Endpoint Manually**
   ```bash
   curl https://10pearlsaqi-production.up.railway.app/health
   ```

**Common causes and fixes:**

#### 1. MongoDB Connection Issues
**Symptoms:** Service crashes on startup, logs show MongoDB connection errors

**Solution:**
- Verify `MONGODB_URI` environment variable is set correctly in Railway
- Check MongoDB Atlas network access (should allow 0.0.0.0/0 or Railway IPs)
- Ensure MongoDB credentials are correct

#### 2. Port Configuration
**Symptoms:** Service starts but Railway can't connect

**Solution:**
- Ensure your app binds to `0.0.0.0` (not `localhost`)
- Use `$PORT` environment variable (Railway sets this automatically)
- Check `railway.toml` start command is correct

#### 3. Cold Start Timeout
**Symptoms:** 502 errors on first request after inactivity

**Solution:**
- Railway has a health check configured (`/health` endpoint)
- The health check timeout is set to 100 seconds in `railway.toml`
- If startup takes longer, increase `healthcheckTimeout`

#### 4. Missing Dependencies
**Symptoms:** Import errors in logs

**Solution:**
- Check `api/requirements.txt` has all dependencies
- Verify Python version matches (should be 3.12)
- Check Railway build logs for installation errors

---

## Debugging Steps

### Step 1: Check Railway Logs

1. Go to [railway.app](https://railway.app)
2. Select your project
3. Click on your service
4. Go to "Logs" tab
5. Look for:
   - Startup errors
   - Import errors
   - Connection errors
   - Timeout errors

### Step 2: Verify Environment Variables

In Railway dashboard → Your Service → Variables:
- `MONGODB_URI` - Should be your MongoDB connection string
- `ALLOWED_ORIGINS` - Should be `*` or your Streamlit URL
- `PORT` - Automatically set by Railway (don't set manually)

### Step 3: Test Health Endpoint

```bash
# Test if service is responding
curl https://10pearlsaqi-production.up.railway.app/health

# Expected response:
# {"status":"healthy","mongodb":"connected","timestamp":"...","service":"AQI Prediction API"}
```

### Step 4: Check Service Configuration

Verify `railway.toml`:
```toml
[deploy]
startCommand = "cd api && uvicorn main:app --host 0.0.0.0 --port $PORT"
healthcheckPath = "/health"
healthcheckTimeout = 100
```

---

## Recent Fixes Applied

### 1. Improved Error Handling
- Added retry logic for 502 errors
- Better error messages in dashboard
- Exponential backoff for retries

### 2. Startup Event
- Added FastAPI startup event to initialize services
- Better error handling during startup
- Logs startup progress

### 3. Health Check Improvements
- Health check no longer crashes if MongoDB is temporarily unavailable
- Returns "degraded" status instead of "unhealthy" for non-critical issues
- Faster response time

### 4. Increased Timeouts
- Prediction requests: 60-90 seconds
- Model listing: 30-45 seconds
- Health checks: 15 seconds

---

## Quick Fixes

### If Service Keeps Crashing:

1. **Check MongoDB Connection**
   ```bash
   # Test MongoDB URI locally
   python -c "from pipelines.mongodb_store import MongoDBStore; store = MongoDBStore(); store._connect(); print('Connected!')"
   ```

2. **Verify Start Command**
   - Should be: `cd api && uvicorn main:app --host 0.0.0.0 --port $PORT`
   - Check in Railway dashboard → Settings → Deploy

3. **Check Python Version**
   - Railway should detect Python 3.12
   - If not, add `runtime.txt` with `python-3.12.0`

4. **Review Dependencies**
   - Ensure `api/requirements.txt` has all needed packages
   - Check for version conflicts

### If Getting 502 Errors:

1. **Wait for Cold Start**
   - First request after inactivity can take 30-60 seconds
   - The dashboard now retries automatically

2. **Check Service Status**
   - Railway dashboard should show service as "Running"
   - If "Crashed", check logs for errors

3. **Verify Health Check**
   - Railway uses `/health` endpoint for health checks
   - Make sure it returns 200 status code

---

## Monitoring

### Railway Dashboard
- **Metrics**: CPU, Memory, Network usage
- **Logs**: Real-time application logs
- **Deployments**: Deployment history and status

### Health Check Endpoint
```bash
# Check service health
curl https://your-api.railway.app/health

# Should return:
{
  "status": "healthy",
  "mongodb": "connected",
  "timestamp": "2024-...",
  "service": "AQI Prediction API"
}
```

---

## Support

If issues persist:
1. Check Railway logs for specific error messages
2. Verify all environment variables are set
3. Test MongoDB connection separately
4. Review recent code changes
5. Check Railway status page for platform issues

---

## Best Practices

1. **Always check logs first** - Most issues are visible in logs
2. **Test health endpoint** - Quick way to verify service is running
3. **Monitor metrics** - Watch CPU/Memory usage
4. **Set up alerts** - Railway can send notifications on failures
5. **Use health checks** - Configure proper health check paths and timeouts
