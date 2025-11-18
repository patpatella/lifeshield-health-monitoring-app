import pandas as pd
import joblib

MODEL_PATH = 'backend/models/isolation_model.joblib'
FEATURES_PATH = 'backend/models/features.joblib'
SCALER_PATH = 'backend/models/scaler.joblib'
NUMERIC_FEATURES_PATH = 'backend/models/numeric_features.joblib'

# Load trained model, features, scaler, numeric_features
model = joblib.load(MODEL_PATH)
features = joblib.load(FEATURES_PATH)        # all model features
scaler = joblib.load(SCALER_PATH)           # fitted scaler
numeric_features = joblib.load(NUMERIC_FEATURES_PATH)  # only columns scaled during training

# Load test data
df = pd.read_csv('data/simulated.csv').head(10)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.set_index('timestamp')

# Add hr_diff
df['hr_diff'] = df['heart_rate'].diff().fillna(0)

# Convert categorical to dummies
if 'activity_level' in df.columns:
    df_cat = pd.get_dummies(df['activity_level'], drop_first=True)
    df = pd.concat([df.drop(columns=['activity_level']), df_cat], axis=1)

# Ensure all features exist
for feat in features:
    if feat not in df.columns:
        df[feat] = 0

# Scale **only numeric features** used in training
df[numeric_features] = scaler.transform(df[numeric_features])

# Reorder columns exactly as training
X = df[features].values

# Predict anomalies
preds = model.predict(X)

# Print results
print("Timestamps:\n", df.index)
print("Predictions:\n", preds.tolist())
