"""Export all features from MongoDB to CSV file."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.mongodb_store import MongoDBStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Export all features to CSV."""
    store = MongoDBStore()
    
    try:
        output_file = store.export_all_to_csv()
        print(f"\nExport completed successfully!")
        print(f"CSV file location: {output_file}")
        print(f"\nYou can also find consolidated CSV at: data/features/features_consolidated.csv")
    except Exception as e:
        print(f"Error exporting to CSV: {str(e)}")
        sys.exit(1)
    finally:
        store.close()

if __name__ == "__main__":
    main()
