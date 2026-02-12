"""Quick start script to set up and test the AQI prediction system."""
import sys
from pathlib import Path
import subprocess

def check_dependencies():
    """Check if required packages are installed."""
    print("Checking dependencies...")
    required_packages = [
        'pandas', 'numpy', 'requests', 'sklearn', 
        'streamlit', 'plotly', 'yaml', 'dotenv'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"[OK] {package}")
        except ImportError:
            missing.append(package)
            print(f"[MISSING] {package}")
    
    if missing:
        print(f"\n[WARNING] Missing packages: {', '.join(missing)}")
        print("Install them with: pip install -r requirements.txt")
        return False
    else:
        print("\n[OK] All dependencies installed!")
        return True

def check_config():
    """Check if configuration files exist."""
    print("\nChecking configuration...")
    project_root = Path(__file__).parent
    
    config_file = project_root / "config" / "config.yaml"
    if config_file.exists():
        print("[OK] config.yaml exists")
    else:
        print("[ERROR] config.yaml missing")
        return False
    
    env_file = project_root / ".env"
    env_example = project_root / "config" / "env.example"
    
    if env_file.exists():
        print("[OK] .env file exists")
    elif env_example.exists():
        print("[WARNING] .env file not found. Copy config/env.example to .env and add your API keys")
    else:
        print("[WARNING] No .env file found")
    
    return True

def create_directories():
    """Create necessary directories."""
    print("\nCreating directories...")
    project_root = Path(__file__).parent
    
    directories = [
        "data/features",
        "data/raw",
        "models",
        "logs"
    ]
    
    for dir_path in directories:
        full_path = project_root / dir_path
        full_path.mkdir(parents=True, exist_ok=True)
        print(f"[OK] {dir_path}")

def main():
    """Main quick start function."""
    print("=" * 50)
    print("AQI Predictor - Quick Start Check")
    print("=" * 50)
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    # Check configuration
    config_ok = check_config()
    
    # Create directories
    create_directories()
    
    print("\n" + "=" * 50)
    if deps_ok and config_ok:
        print("[SUCCESS] Setup complete! You can now:")
        print("  1. Run feature pipeline: python pipelines/feature_pipeline.py")
        print("  2. Train models: python pipelines/training_pipeline.py")
        print("  3. Launch dashboard: python scripts/run_dashboard.py")
        print("  4. Start API: cd api && uvicorn main:app --host 0.0.0.0 --port 8000")
    else:
        print("[WARNING] Setup incomplete. Please fix the issues above.")
    print("=" * 50)

if __name__ == "__main__":
    main()
