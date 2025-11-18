# backend/train_isolation.py
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
from backend.services.preprocess import preprocess
import os

DATA_PATH = 'data/simulated.csv'
MODEL_PATH = 'backend/models/isolation_model.joblib'
SCALER_PATH = 'backend/models/scaler.joblib'
FEATURES_PATH = 'backend/models/features.joblib'

def train_and_save(contamination=0.02):
    # Load dataset
    df = pd.read_csv(DATA_PATH)

    # Preprocess data (this will also save scaler, features, numeric_features)
    df_proc, scaler, features = preprocess(df, save_scaler_path=SCALER_PATH)

    # Load numeric_features saved by preprocess.py
    numeric_features_path = os.path.join(os.path.dirname(SCALER_PATH), 'numeric_features.joblib')
    numeric_features = joblib.load(numeric_features_path)
    print(f"Numeric features used for scaling: {numeric_features}")

    # Prepare training data
    X = df_proc[features].values

    # Train IsolationForest
    model = IsolationForest(contamination=contamination, random_state=42)
    model.fit(X)

    # Save model and features
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(features, FEATURES_PATH)

    print(f"Saved IsolationForest to {MODEL_PATH}")
    return model, features

if __name__ == '__main__':
    train_and_save(contamination=0.02)
