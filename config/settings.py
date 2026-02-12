"""Configuration settings loader."""
import os
import yaml
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get project root directory
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"


def load_config():
    """Load configuration from YAML file."""
    with open(CONFIG_PATH, 'r') as f:
        config = yaml.safe_load(f)
    
    # Override with environment variables if present
    if os.getenv('OPENMETEO_API_KEY'):
        if 'openmeteo' not in config.get('apis', {}):
            config['apis']['openmeteo'] = {}
        config['apis']['openmeteo']['api_key'] = os.getenv('OPENMETEO_API_KEY')
    if os.getenv('MONGODB_URI'):
        config['mongodb']['connection_string'] = os.getenv('MONGODB_URI')
    
    return config


# Global config instance
config = load_config()
