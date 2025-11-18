# backend/services/preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os

def preprocess(df, save_scaler_path='backend/models/scaler.joblib'):
    """
    Preprocess the health monitoring dataset:
    - Resample numeric columns
    - Handle categorical features with dummies
    - Add hr_diff feature
    - Scale numeric features
    - Save scaler, features, and numeric_features
    """
    df = df.copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')

    # Define numeric and categorical columns
    numeric_cols = ['heart_rate', 'blood_oxygen']
    cat_cols = ['activity_level'] if 'activity_level' in df.columns else []

    # Resample numeric columns only (1-minute intervals)
    df_numeric = df[numeric_cols].resample('1min').mean().ffill()

    # Handle categorical columns: take mode in each resample
    df_cat = pd.DataFrame(index=df_numeric.index)
    for col in cat_cols:
        df_cat[col] = df[col].resample('1min').agg(lambda x: x.mode()[0] if not x.mode().empty else 'low')

    # Combine numeric + categorical
    df_proc = pd.concat([df_numeric, df_cat], axis=1)

    # Add hr_diff feature
    df_proc['hr_diff'] = df_proc['heart_rate'].diff().fillna(0)

    # Convert categorical to dummies
    if cat_cols:
        df_proc = pd.get_dummies(df_proc, columns=cat_cols, drop_first=True)

    # List of features to use in the model (numeric + dummies)
    features = list(df_proc.columns)

    # Identify numeric features for scaling
    numeric_features = numeric_cols + ['hr_diff']

    # Scale numeric features only
    scaler = StandardScaler()
    df_proc[numeric_features] = scaler.fit_transform(df_proc[numeric_features])

    # Create folder if not exists
    os.makedirs(os.path.dirname(save_scaler_path), exist_ok=True)

    # Save scaler
    joblib.dump(scaler, save_scaler_path)

    # Save features list for model
    features_path = os.path.join(os.path.dirname(save_scaler_path), 'features.joblib')
    joblib.dump(features, features_path)

    # Save numeric features used for scaling
    numeric_features_path = os.path.join(os.path.dirname(save_scaler_path), 'numeric_features.joblib')
    joblib.dump(numeric_features, numeric_features_path)

    return df_proc, scaler, features
