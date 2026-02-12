"""Script to check data quality and verify row/column counts."""
import pandas as pd
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def check_features_data():
    """Check features.csv data quality."""
    features_file = project_root / "data" / "features" / "features.csv"
    raw_file = project_root / "data" / "raw" / "raw_data.csv"
    
    print("=" * 70)
    print("DATA QUALITY CHECK")
    print("=" * 70)
    
    # Check features
    if features_file.exists():
        df_features = pd.read_csv(features_file)
        df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
        
        print(f"\n📊 FEATURES DATA (features.csv)")
        print(f"  Total Records: {len(df_features):,}")
        print(f"  Total Columns: {len(df_features.columns)}")
        print(f"\n  Date Range:")
        print(f"    Start: {df_features['timestamp'].min()}")
        print(f"    End: {df_features['timestamp'].max()}")
        
        days_span = (df_features['timestamp'].max() - df_features['timestamp'].min()).days
        print(f"    Days Span: {days_span}")
        
        # Expected records
        unique_dates = df_features['timestamp'].dt.date.nunique()
        expected_records = unique_dates * 24
        print(f"\n  Expected Records:")
        print(f"    Unique Dates: {unique_dates}")
        print(f"    Expected (dates × 24 hours): {expected_records:,}")
        print(f"    Actual: {len(df_features):,}")
        print(f"    Difference: {len(df_features) - expected_records:,}")
        
        # Check for duplicates
        duplicate_timestamps = df_features.duplicated(subset=['timestamp']).sum()
        unique_timestamps = df_features['timestamp'].nunique()
        print(f"\n  Duplicate Check:")
        print(f"    Unique Timestamps: {unique_timestamps:,}")
        print(f"    Duplicate Timestamps: {duplicate_timestamps:,}")
        
        # Records per day
        records_per_day = len(df_features) / unique_dates if unique_dates > 0 else 0
        print(f"    Avg Records per Day: {records_per_day:.2f} (expected: 24)")
        
        # Check for missing values
        missing_pct = (df_features.isna().sum() / len(df_features) * 100).round(2)
        high_missing = missing_pct[missing_pct > 50]
        print(f"\n  Missing Values:")
        print(f"    Columns with >50% missing: {len(high_missing)}")
        if len(high_missing) > 0:
            print(f"    {list(high_missing.index)}")
        
    else:
        print("\n❌ features.csv not found!")
    
    # Check raw data
    if raw_file.exists():
        df_raw = pd.read_csv(raw_file)
        df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
        
        print(f"\n📊 RAW DATA (raw_data.csv)")
        print(f"  Total Records: {len(df_raw):,}")
        print(f"  Total Columns: {len(df_raw.columns)}")
        print(f"  Date Range: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
        
        raw_unique_dates = df_raw['timestamp'].dt.date.nunique()
        print(f"  Unique Dates: {raw_unique_dates}")
        
    else:
        print("\n❌ raw_data.csv not found!")
    
    # Comparison
    if features_file.exists() and raw_file.exists():
        print(f"\n📊 COMPARISON")
        print(f"  Raw Records: {len(df_raw):,}")
        print(f"  Feature Records: {len(df_features):,}")
        print(f"  Difference: {len(df_features) - len(df_raw):,}")
        
        if len(df_features) > len(df_raw):
            print(f"  ⚠️  Features have MORE records than raw data")
            print(f"     This suggests duplicate processing or multiple runs")
        elif len(df_features) < len(df_raw):
            print(f"  ⚠️  Features have FEWER records than raw data")
            print(f"     Some records may have been dropped during processing")
        else:
            print(f"  ✅ Record counts match")
    
    # Assessment
    print(f"\n" + "=" * 70)
    print("ASSESSMENT")
    print("=" * 70)
    
    if features_file.exists():
        expected = unique_dates * 24
        actual = len(df_features)
        diff = actual - expected
        
        if abs(diff) <= 50:
            print(f"✅ Row count is GOOD")
            print(f"   {actual:,} records is close to expected {expected:,} (±50 tolerance)")
        elif diff > 50:
            print(f"⚠️  Row count is HIGHER than expected")
            print(f"   {actual:,} records vs expected {expected:,} (+{diff:,} extra)")
            print(f"   Possible causes:")
            print(f"   - Duplicate records from multiple runs")
            print(f"   - Overlapping date ranges in backfill")
            print(f"   - Multiple feature pipeline runs")
            print(f"   Recommendation: Check for duplicates and clean if needed")
        else:
            print(f"⚠️  Row count is LOWER than expected")
            print(f"   {actual:,} records vs expected {expected:,} ({abs(diff):,} missing)")
            print(f"   Possible causes:")
            print(f"   - Incomplete backfill")
            print(f"   - Data dropped during cleaning")
            print(f"   Recommendation: Re-run backfill for missing dates")
        
        if len(df_features.columns) == 69:
            print(f"\n✅ Column count is CORRECT")
            print(f"   69 columns matches expected feature count")
        else:
            print(f"\n⚠️  Column count is UNEXPECTED")
            print(f"   {len(df_features.columns)} columns vs expected 69")
            print(f"   Check feature engineering pipeline")
    
    print(f"\n" + "=" * 70)

if __name__ == "__main__":
    check_features_data()
