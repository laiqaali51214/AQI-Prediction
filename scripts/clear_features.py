"""Clear feature data from MongoDB (use with caution)."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.mongodb_store import MongoDBStore
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def clear_all_features():
    """Clear all features from MongoDB."""
    store = MongoDBStore()
    
    try:
        if store.db is None:
            logger.error("MongoDB connection not established")
            return False
        
        collection = store.db[store.features_collection_name]
        
        # Count existing records
        count = collection.count_documents({})
        logger.info(f"Found {count} feature records in MongoDB")
        
        if count == 0:
            logger.info("No records to delete")
            return True
        
        # Delete all records
        result = collection.delete_many({})
        logger.info(f"Deleted {result.deleted_count} feature records from MongoDB")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing features: {str(e)}")
        return False
    finally:
        store.close()

def clear_constant_aqi_features():
    """Clear features with constant AQI values (AQI = 150.0)."""
    store = MongoDBStore()
    
    try:
        if store.db is None:
            logger.error("MongoDB connection not established")
            return False
        
        collection = store.db[store.features_collection_name]
        
        # Find records with AQI = 150.0
        query = {'aqi': 150.0}
        count = collection.count_documents(query)
        logger.info(f"Found {count} records with AQI = 150.0")
        
        if count == 0:
            logger.info("No constant AQI records to delete")
            return True
        
        # Delete records with constant AQI
        result = collection.delete_many(query)
        logger.info(f"Deleted {result.deleted_count} records with constant AQI")
        
        return True
    except Exception as e:
        logger.error(f"Error clearing constant AQI features: {str(e)}")
        return False
    finally:
        store.close()

def deduplicate_features():
    """Remove duplicate records from MongoDB, keeping the most recent one for each timestamp."""
    store = MongoDBStore()
    
    try:
        if store.db is None:
            logger.error("MongoDB connection not established")
            return False
        
        collection = store.db[store.features_collection_name]
        
        # Count total records before deduplication
        total_before = collection.count_documents({})
        logger.info(f"Total records before deduplication: {total_before:,}")
        
        # Get all unique timestamps and their document IDs
        pipeline = [
            {
                '$group': {
                    '_id': '$timestamp',
                    'count': {'$sum': 1},
                    'ids': {'$push': '$_id'},
                    'max_inserted_at': {'$max': '$inserted_at'}
                }
            },
            {
                '$match': {
                    'count': {'$gt': 1}
                }
            }
        ]
        
        duplicates = list(collection.aggregate(pipeline))
        duplicate_count = len(duplicates)
        total_duplicates = sum(doc['count'] - 1 for doc in duplicates)
        
        logger.info(f"Found {duplicate_count:,} unique timestamps with duplicates")
        logger.info(f"Total duplicate records to remove: {total_duplicates:,}")
        
        if total_duplicates == 0:
            logger.info("No duplicates found. Collection is clean.")
            return True
        
        # Remove duplicates, keeping the most recent document (by inserted_at or _id)
        removed_count = 0
        for dup in duplicates:
            ids = dup['ids']
            if len(ids) > 1:
                # Keep the one with the latest inserted_at, or the last _id if no inserted_at
                ids_to_remove = ids[:-1]  # Remove all except the last one
                result = collection.delete_many({'_id': {'$in': ids_to_remove}})
                removed_count += result.deleted_count
        
        # Count total records after deduplication
        total_after = collection.count_documents({})
        
        logger.info(f"Removed {removed_count:,} duplicate records")
        logger.info(f"Total records after deduplication: {total_after:,}")
        logger.info(f"Reduction: {total_before - total_after:,} records ({((total_before - total_after) / total_before * 100):.1f}%)")
        
        return True
        
    except Exception as e:
        logger.error(f"Error deduplicating features: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        store.close()

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Clear or deduplicate feature data from MongoDB')
    parser.add_argument('--all', action='store_true', help='Clear all features')
    parser.add_argument('--constant-aqi', action='store_true', 
                       help='Clear only features with constant AQI (150.0)')
    parser.add_argument('--deduplicate', action='store_true',
                       help='Remove duplicate records, keeping most recent for each timestamp')
    
    args = parser.parse_args()
    
    if args.all:
        print("WARNING: This will delete ALL feature records from MongoDB!")
        confirm = input("Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            clear_all_features()
        else:
            print("Operation cancelled")
    elif args.constant_aqi:
        print("Clearing features with constant AQI (150.0)...")
        clear_constant_aqi_features()
    elif args.deduplicate:
        print("Removing duplicate records from MongoDB...")
        print("This will keep the most recent record for each timestamp.")
        confirm = input("Type 'yes' to proceed: ")
        if confirm.lower() == 'yes':
            deduplicate_features()
        else:
            print("Operation cancelled")
    else:
        print("Usage:")
        print("  python scripts/clear_features.py --deduplicate  # Remove duplicates (recommended)")
        print("  python scripts/clear_features.py --constant-aqi  # Clear constant AQI records")
        print("  python scripts/clear_features.py --all           # Clear ALL records (requires confirmation)")

if __name__ == "__main__":
    main()
