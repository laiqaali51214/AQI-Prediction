"""Helper script to run optimized backfill from a specific date."""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from pipelines.backfill import backfill_historical_data


def get_last_date_from_csv():
    """Get the last date from raw_data.csv to resume from."""
    raw_data_file = project_root / "data" / "raw" / "raw_data.csv"
    
    if not raw_data_file.exists():
        return None
    
    try:
        import pandas as pd
        df = pd.read_csv(raw_data_file)
        
        if 'timestamp' in df.columns and len(df) > 0:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            last_date = df['timestamp'].max()
            return last_date.date()
    except Exception as e:
        print(f"Error reading CSV: {e}")
    
    return None


def main():
    """Run optimized backfill, optionally resuming from last date."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Run optimized backfill script')
    parser.add_argument('--start-date', type=str, default=None,
                       help='Start date in YYYY-MM-DD format (default: last date in CSV + 1 day)')
    parser.add_argument('--end-date', type=str, default=None,
                       help='End date in YYYY-MM-DD format (default: today)')
    parser.add_argument('--batch-days', type=int, default=30,
                       help='Number of days to fetch per API call (default: 30)')
    parser.add_argument('--resume', action='store_true',
                       help='Resume from last date in raw_data.csv')
    
    args = parser.parse_args()
    
    # Determine start date
    if args.resume:
        last_date = get_last_date_from_csv()
        if last_date:
            start_date = (last_date + timedelta(days=1)).strftime('%Y-%m-%d')
            print(f"Resuming from last date in CSV: {last_date}")
            print(f"Starting backfill from: {start_date}")
        else:
            print("No existing data found. Starting from 1 year ago.")
            # Use yesterday as end date (exclude current day)
            end_date = datetime.now() - timedelta(days=1)
            start_date = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
    elif args.start_date:
        start_date = args.start_date
    else:
        # Default: 1 year ago
        # Use yesterday as end date (exclude current day)
        end_date = datetime.now() - timedelta(days=1)
        start_date = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
    
    # Determine end date (default to yesterday, exclude current day)
    if args.end_date:
        end_date = args.end_date
    else:
        # Use yesterday as default end date (exclude current day)
        end_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    print(f"\n{'='*60}")
    print(f"Optimized Backfill Script")
    print(f"{'='*60}")
    print(f"Start Date: {start_date}")
    print(f"End Date: {end_date}")
    print(f"Batch Size: {args.batch_days} days")
    print(f"{'='*60}\n")
    
    # Run backfill
    backfill_historical_data(start_date, end_date, args.batch_days)


if __name__ == "__main__":
    main()
