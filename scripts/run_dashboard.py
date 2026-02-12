"""Script to run the Streamlit dashboard."""
import subprocess
import sys
from pathlib import Path

if __name__ == "__main__":
    # Get project root (parent of scripts directory)
    project_root = Path(__file__).parent.parent
    dashboard_path = project_root / "app" / "dashboard.py"
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard_path)])
