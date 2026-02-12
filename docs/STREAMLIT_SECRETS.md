# Streamlit Secrets Management Guide

This guide explains how to use Streamlit's built-in secrets management for secure configuration.

## Why Use Streamlit Secrets?

✅ **More Secure**: Encrypted and stored securely by Streamlit  
✅ **Easy to Manage**: Centralized in Streamlit Cloud dashboard  
✅ **No Code Changes**: Access via `st.secrets` (same API)  
✅ **Better UX**: No need to manage environment variables separately  

---

## For Streamlit Cloud Deployment

### Step 1: Add Secrets in Streamlit Cloud

1. Go to your app on [Streamlit Cloud](https://share.streamlit.io)
2. Click the **"⋮"** menu (three dots) → **"Settings"**
3. Go to the **"Secrets"** tab
4. Add your secrets in TOML format:

```toml
FASTAPI_URL = "https://10pearlsaqi-production.up.railway.app"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
```

5. Click **"Save"**

### Step 2: Access Secrets in Code

The dashboard code automatically uses Streamlit secrets:

```python
# In app/dashboard.py
try:
    API_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")
except (AttributeError, FileNotFoundError, KeyError):
    # Fallback to environment variable for local development
    API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
```

---

## For Local Development

### Option 1: Create Local Secrets File (Recommended)

1. Create `.streamlit/secrets.toml` (this file is gitignored):

```toml
FASTAPI_URL = "http://localhost:8000"
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
```

2. The code will automatically use these secrets when running locally

### Option 2: Use Environment Variables

If you don't create `secrets.toml`, the code falls back to environment variables:

```bash
# Windows PowerShell
$env:FASTAPI_URL="http://localhost:8000"
$env:MONGODB_URI="mongodb+srv://..."

# Linux/Mac
export FASTAPI_URL="http://localhost:8000"
export MONGODB_URI="mongodb+srv://..."
```

---

## Secrets Structure

### Current Secrets Used

```toml
# FastAPI Backend URL
FASTAPI_URL = "https://your-api.railway.app"

# MongoDB Connection String (optional - only if dashboard needs direct DB access)
MONGODB_URI = "mongodb+srv://username:password@cluster.mongodb.net/database?retryWrites=true&w=majority"
```

### Adding More Secrets

You can add any secrets you need:

```toml
FASTAPI_URL = "https://your-api.railway.app"
MONGODB_URI = "mongodb+srv://..."
API_KEY = "your-api-key"
CUSTOM_SETTING = "value"
```

Access them in code:
```python
api_key = st.secrets.get("API_KEY")
custom = st.secrets.get("CUSTOM_SETTING")
```

---

## Security Best Practices

### ✅ DO:
- Use Streamlit secrets for sensitive data (API keys, passwords, connection strings)
- Keep `.streamlit/secrets.toml` in `.gitignore` (already done)
- Use different secrets for development and production
- Rotate secrets regularly

### ❌ DON'T:
- Commit secrets to Git
- Share secrets publicly
- Use the same secrets across environments
- Hardcode secrets in your code

---

## Troubleshooting

### Problem: `st.secrets` not found

**Solution**: Make sure you're using Streamlit 1.10.0 or later:
```bash
pip install streamlit>=1.10.0
```

### Problem: Secrets not loading in Streamlit Cloud

**Solution**: 
1. Check that secrets are saved in the Streamlit Cloud dashboard
2. Verify TOML syntax is correct (no quotes around keys)
3. Redeploy the app after adding secrets

### Problem: Local secrets not working

**Solution**:
1. Ensure `.streamlit/secrets.toml` exists in your project root
2. Check file permissions (should be readable)
3. Verify TOML syntax is correct
4. Restart Streamlit after creating/updating secrets

---

## Example: Complete Setup

### 1. Streamlit Cloud (Production)

In Streamlit Cloud dashboard → Settings → Secrets:
```toml
FASTAPI_URL = "https://10pearlsaqi-production.up.railway.app"
MONGODB_URI = "mongodb+srv://prod-user:prod-pass@cluster.mongodb.net/aqi_predictor?retryWrites=true&w=majority"
```

### 2. Local Development

Create `.streamlit/secrets.toml`:
```toml
FASTAPI_URL = "http://localhost:8000"
MONGODB_URI = "mongodb+srv://dev-user:dev-pass@cluster.mongodb.net/aqi_predictor_dev?retryWrites=true&w=majority"
```

### 3. Code (No Changes Needed)

The dashboard code automatically handles both:
```python
# Works in both Streamlit Cloud and local development
try:
    API_URL = st.secrets.get("FASTAPI_URL", "http://localhost:8000")
except (AttributeError, FileNotFoundError, KeyError):
    API_URL = os.getenv("FASTAPI_URL", "http://localhost:8000")
```

---

## Quick Reference

### Access Secrets
```python
# Get a secret (with default fallback)
value = st.secrets.get("KEY", "default_value")

# Get a secret (raises error if not found)
value = st.secrets["KEY"]

# Check if secret exists
if "KEY" in st.secrets:
    value = st.secrets["KEY"]
```

### Streamlit Cloud
- Dashboard: [share.streamlit.io](https://share.streamlit.io)
- Settings → Secrets tab
- Format: TOML

### Local Development
- File: `.streamlit/secrets.toml`
- Format: TOML
- Gitignored: ✅ Yes (already in `.gitignore`)

---

## Migration from Environment Variables

If you're currently using environment variables, migration is easy:

1. **In Streamlit Cloud**: Add secrets in the dashboard (Settings → Secrets)
2. **Locally**: Create `.streamlit/secrets.toml` with the same values
3. **Code**: Already updated to use `st.secrets` with fallback to environment variables

No breaking changes - environment variables still work as fallback!
