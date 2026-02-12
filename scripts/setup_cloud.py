"""Interactive cloud setup script for AQI Predictor."""
import os
import sys
from pathlib import Path
import json
import yaml

def print_header(text):
    """Print a formatted header."""
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60 + "\n")

def print_step(step_num, description):
    """Print a step description."""
    print(f"\n[STEP {step_num}] {description}")
    print("-" * 60)

def get_user_input(prompt, default=None, required=True):
    """Get user input with optional default."""
    if default:
        prompt_text = f"{prompt} (default: {default}): "
    else:
        prompt_text = f"{prompt}: "
    
    while True:
        value = input(prompt_text).strip()
        if value:
            return value
        elif default:
            return default
        elif not required:
            return ""
        else:
            print("This field is required. Please enter a value.")

def setup_api_keys():
    """Step 1: Set up API keys."""
    print_step(1, "API Keys Configuration")
    
    print("\nWe need API keys and database configuration for the following services:")
    print("1. AQICN API - For air quality data")
    print("2. OpenWeather API - For weather data")
    print("3. MongoDB Atlas - For feature and model storage")
    
    api_keys = {}
    
    print("\n--- AQICN API ---")
    print("Get your token from: https://aqicn.org/api/")
    print("Sign up at: https://aqicn.org/data-platform/token/")
    aqicn_token = get_user_input("Enter AQICN API Token", required=True)
    api_keys['AQICN_TOKEN'] = aqicn_token
    
    print("\n--- OpenWeather API ---")
    print("Get your API key from: https://openweathermap.org/api")
    print("Sign up at: https://home.openweathermap.org/users/sign_up")
    openweather_token = get_user_input("Enter OpenWeather API Key", required=True)
    api_keys['OPENWEATHER_TOKEN'] = openweather_token
    
    print("\n--- MongoDB Atlas ---")
    print("MongoDB Atlas provides cloud database for features and models")
    print("Sign up at: https://www.mongodb.com/cloud/atlas")
    print("After creating cluster, get connection string from: Connect -> Connect your application")
    mongodb_uri = get_user_input("Enter MongoDB Connection String (MONGODB_URI)", required=True)
    api_keys['MONGODB_URI'] = mongodb_uri
    
    return api_keys

def create_env_file(api_keys):
    """Create .env file with API keys."""
    print_step(2, "Creating .env file")
    
    env_path = Path(".env")
    
    if env_path.exists():
        overwrite = input("\n.env file already exists. Overwrite? (y/n): ").strip().lower()
        if overwrite != 'y':
            print("Keeping existing .env file")
            return
    
    with open(env_path, 'w') as f:
        f.write("# API Keys\n")
        f.write(f"AQICN_TOKEN={api_keys['AQICN_TOKEN']}\n")
        f.write(f"OPENWEATHER_TOKEN={api_keys['OPENWEATHER_TOKEN']}\n")
        f.write(f"\n# MongoDB Configuration\n")
        f.write(f"MONGODB_URI={api_keys['MONGODB_URI']}\n")
    
    print(f"[OK] Created .env file at {env_path.absolute()}")
    print("[INFO] Make sure .env is in .gitignore (it should be already)")

def setup_city_config():
    """Step 3: Configure city settings."""
    print_step(3, "City Configuration")
    
    print("\nEnter your city details for AQI prediction:")
    
    city_name = get_user_input("City Name", default="New York")
    latitude = float(get_user_input("Latitude", default="40.7128"))
    longitude = float(get_user_input("Longitude", default="-74.0060"))
    
    print("\nCommon timezones:")
    print("  - America/New_York (Eastern)")
    print("  - America/Chicago (Central)")
    print("  - America/Denver (Mountain)")
    print("  - America/Los_Angeles (Pacific)")
    print("  - Europe/London")
    print("  - Asia/Kolkata")
    timezone = get_user_input("Timezone", default="America/New_York")
    
    # Update config.yaml
    config_path = Path("config/config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    config['city']['name'] = city_name
    config['city']['latitude'] = latitude
    config['city']['longitude'] = longitude
    config['city']['timezone'] = timezone
    
    with open(config_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"[OK] Updated config/config.yaml with city: {city_name}")

def setup_github_actions():
    """Step 4: Set up GitHub Actions secrets."""
    print_step(4, "GitHub Actions Setup")
    
    print("\nTo enable GitHub Actions CI/CD, you need to add secrets to your repository:")
    print("\n1. Go to your GitHub repository")
    print("2. Click on 'Settings' → 'Secrets and variables' → 'Actions'")
    print("3. Click 'New repository secret'")
    print("\nAdd the following secrets:")
    print("\n  Name: AQICN_TOKEN")
    print("  Value: [Your AQICN token]")
    print("\n  Name: OPENWEATHER_TOKEN")
    print("  Value: [Your OpenWeather API key]")
    print("\n  Name: MONGODB_URI")
    print("  Value: [Your MongoDB connection string]")
    
    print("\n[INFO] After adding secrets, GitHub Actions will automatically:")
    print("  - Run feature pipeline every hour")
    print("  - Run training pipeline daily at 2 AM UTC")
    
    input("\nPress Enter when you've added the secrets (or if you'll do it later)...")

def test_api_connections(api_keys):
    """Step 5: Test API connections."""
    print_step(5, "Testing API Connections")
    
    print("\nTesting API connections...")
    
    try:
        import requests
    except ImportError:
        print("[WARNING] 'requests' package not installed. Skipping API tests.")
        print("Install with: pip install requests")
        return
    
    # Test AQICN
    try:
        url = "https://api.waqi.info/feed/geo:40.7128;-74.0060/"
        params = {'token': api_keys['AQICN_TOKEN']}
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if data.get('status') == 'ok':
                print("[OK] AQICN API: Connected successfully")
            else:
                print(f"[ERROR] AQICN API: Error - {data.get('data', 'Unknown error')}")
        else:
            print(f"[ERROR] AQICN API: HTTP {response.status_code}")
    except Exception as e:
        print(f"[ERROR] AQICN API: Connection failed - {str(e)}")
    
    # Test OpenWeather
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            'lat': 40.7128,
            'lon': -74.0060,
            'appid': api_keys['OPENWEATHER_TOKEN'],
            'units': 'metric'
        }
        response = requests.get(url, params=params, timeout=10)
        
        if response.status_code == 200:
            print("[OK] OpenWeather API: Connected successfully")
        else:
            print(f"[ERROR] OpenWeather API: HTTP {response.status_code}")
            if response.status_code == 401:
                print("  (Invalid API key)")
    except Exception as e:
        print(f"[ERROR] OpenWeather API: Connection failed - {str(e)}")

def create_setup_summary(api_keys):
    """Create a setup summary file."""
    summary = {
        "setup_completed": True,
        "apis_configured": {
            "aqicn": bool(api_keys.get('AQICN_TOKEN')),
            "openweather": bool(api_keys.get('OPENWEATHER_TOKEN')),
            "mongodb": bool(api_keys.get('MONGODB_URI'))
        },
        "next_steps": [
            "Run: python pipelines/feature_pipeline.py",
            "Backfill data: python pipelines/backfill.py --start-date 2024-01-01 --end-date 2024-01-31",
            "Train models: python pipelines/training_pipeline.py",
            "Launch dashboard: python run_dashboard.py"
        ]
    }
    
    with open("setup_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n[OK] Setup summary saved to setup_summary.json")

def main():
    """Main setup function."""
    print_header("AQI Predictor - Cloud Setup Wizard")
    
    print("This wizard will help you set up:")
    print("  [*] API keys for data sources")
    print("  [*] City configuration")
    print("  [*] GitHub Actions secrets")
    print("  [*] Test API connections")
    
    input("\nPress Enter to start...")
    
    # Step 1: API Keys
    api_keys = setup_api_keys()
    
    # Step 2: Create .env file
    create_env_file(api_keys)
    
    # Step 3: City configuration
    setup_city_config()
    
    # Step 4: GitHub Actions
    setup_github_actions()
    
    # Step 5: Test connections
    test_apis = input("\nDo you want to test API connections now? (y/n): ").strip().lower()
    if test_apis == 'y':
        test_api_connections(api_keys)
    
    # Create summary
    create_setup_summary(api_keys)
    
    print_header("Setup Complete!")
    print("\nNext steps:")
    print("1. Verify .env file contains your API keys")
    print("2. Add GitHub secrets if using CI/CD")
    print("3. Run: python pipelines/feature_pipeline.py")
    print("4. Train models: python pipelines/training_pipeline.py")
    print("5. Start API: cd api && uvicorn main:app --host 0.0.0.0 --port 8000")
    print("6. Launch dashboard: python scripts/run_dashboard.py")
    print("\nFor detailed instructions, see docs/SETUP_INSTRUCTIONS.md")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nSetup cancelled by user.")
        sys.exit(0)
    except Exception as e:
        print(f"\n\nError during setup: {str(e)}")
        sys.exit(1)
