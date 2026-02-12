"""Check for overfitting issues in the model."""
import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
from pipelines.mongodb_store import MongoDBStore
from pipelines.training_pipeline import ModelTrainer

def main():
    store = MongoDBStore()
    
    # Load features
    df = store.get_features()
    print(f"Total samples: {len(df)}")
    
    if 'aqi' in df.columns:
        print(f"\nAQI Statistics:")
        print(f"  Unique values: {df['aqi'].nunique()}")
        print(f"  Mean: {df['aqi'].mean():.2f}")
        print(f"  Std: {df['aqi'].std():.2f}")
        print(f"  Min: {df['aqi'].min():.2f}")
        print(f"  Max: {df['aqi'].max():.2f}")
        print(f"\nTop 10 AQI values:")
        print(df['aqi'].value_counts().head(10))
    
    # Check for lag features that might leak information
    lag_features = [c for c in df.columns if 'lag_24' in c or 'lag_24h' in c]
    print(f"\nLag 24h features (potential data leakage): {lag_features}")
    
    # Check target creation
    trainer = ModelTrainer()
    X_train, X_test, y_train, y_test, feature_cols = trainer.prepare_data(df)
    
    print(f"\nTraining Data:")
    print(f"  Samples: {len(X_train)}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Target unique values: {y_train.nunique()}")
    print(f"  Target std: {y_train.std():.2f}")
    
    print(f"\nTest Data:")
    print(f"  Samples: {len(X_test)}")
    print(f"  Target unique values: {y_test.nunique()}")
    print(f"  Target std: {y_test.std():.2f}")
    
    # Check if target is constant
    if y_train.nunique() == 1:
        print("\nWARNING: Target has only 1 unique value - model will predict constant!")
    elif y_train.std() < 0.01:
        print("\nWARNING: Target has very low variance - model will overfit!")
    
    # Check for features that directly correlate with target
    print(f"\nFeature columns used: {feature_cols[:10]}...")
    
    # Check if lag_24h is in features (this would be data leakage)
    if any('lag_24' in col for col in feature_cols):
        print("\nCRITICAL: lag_24h features are included - this is DATA LEAKAGE!")
        print("The target is AQI shifted 24h forward, so lag_24h is the target itself!")
    
    store.close()

if __name__ == "__main__":
    main()
