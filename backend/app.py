# backend/app.py
from flask import Flask, request, jsonify, send_file
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# -------------------------------
# Paths to saved objects
# -------------------------------
BASE_MODEL_PATH = 'data/bidmc/models'
ISO_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_iso_model.joblib')
RF_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_rf_model.joblib')
SCALER_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_scaler.joblib')
FEATURES_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_features.pkl')
DISEASE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'disease_rf_model.joblib')

# Load trained objects
iso_model = joblib.load(ISO_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
disease_model = joblib.load(DISEASE_MODEL_PATH)

# Identify numeric features for scaling
numeric_features = [col for col in features if any(k in col.upper() for k in ['HR', 'SPO2', 'BREATH', 'ACTIVITY'])]

# -------------------------------
# Logging setup
# -------------------------------
LOG_FILE = os.path.join(BASE_MODEL_PATH, 'health_predictions_log.csv')
if not os.path.exists(LOG_FILE):
    log_df = pd.DataFrame(columns=[
        'timestamp', 'record_index', 'rf_anomaly', 'iso_anomaly', 
        'disease_risk', 'lifestyle_recommendations', 'activity_level'
    ])
    log_df.to_csv(LOG_FILE, index=False)

# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)

@app.route('/')
def index():
    return "AI Health Monitoring API is running!"

# -------------------------------
# Combined AI Health Engine with Logging
# -------------------------------
@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json()
    if not data:
        return jsonify({"error": "No input data provided"}), 400

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return jsonify({"error": "Input must be a dict or a list of health records"}), 400

    df = pd.DataFrame(data)
    anomaly_results = preprocess_and_predict(df)
    disease_results = preprocess_disease(df)

    combined_results = []
    log_entries = []
    timestamp_now = datetime.now().isoformat()

    for i in range(len(df)):
        combined = {
            "record_index": i,
            "timestamp": df['timestamp'].iloc[i] if 'timestamp' in df.columns else timestamp_now,
            "heart_rate": df['heart_rate'].iloc[i] if 'heart_rate' in df.columns else 0,
            "SPO2": df['SPO2'].iloc[i] if 'SPO2' in df.columns else 0,
            "activity_level": df['activity_level'].iloc[i] if 'activity_level' in df.columns else 'moderate',
            "anomaly_rf": anomaly_results['rf_predictions'][i],
            "anomaly_iso": anomaly_results['iso_predictions'][i],
            "disease_risk": disease_results['disease_risk_prediction'][i],
            "lifestyle_recommendations": disease_results['lifestyle_recommendations'][i]
        }
        combined_results.append(combined)

        log_entries.append({
            "timestamp": combined["timestamp"],
            "record_index": i,
            "rf_anomaly": combined['anomaly_rf'],
            "iso_anomaly": combined['anomaly_iso'],
            "disease_risk": combined['disease_risk'],
            "lifestyle_recommendations": "|".join(combined['lifestyle_recommendations']),
            "activity_level": combined['activity_level']
        })

    log_df = pd.DataFrame(log_entries)
    log_df.to_csv(LOG_FILE, mode='a', header=False, index=False)

    return jsonify({"results": combined_results})

# -------------------------------
# Get historical prediction logs
# -------------------------------
@app.route('/get_logs', methods=['GET'])
def get_logs():
    log_format = request.args.get('format', 'json').lower()
    if not os.path.exists(LOG_FILE):
        return jsonify({"error": "No logs available"}), 404

    if log_format == 'csv':
        return send_file(LOG_FILE, mimetype='text/csv', as_attachment=True, download_name='health_predictions_log.csv')
    else:
        df = pd.read_csv(LOG_FILE)
        return jsonify(df.to_dict(orient='records'))

# -------------------------------
# Disease Prediction Helper (FIXED)
# -------------------------------

def preprocess_disease(df):
    df = df.copy()

    # ---------------------------------
    # Compute missing engineered features
    # ---------------------------------
    df["HR"] = df["heart_rate"]
    df["SPO2"] = df["SPO2"]

    # Sliding-window / fallback logic for single-row inputs
    df["HR_MEAN"] = df["HR"].rolling(5, min_periods=1).mean()
    df["HR_STD"] = df["HR"].rolling(5, min_periods=1).std().fillna(0)
    df["HR_DIFF"] = df["HR"].diff().fillna(0)
    df["HR_VAR"] = df["HR"].rolling(5, min_periods=1).var().fillna(0)

    df["SPO2_DIFF"] = df["SPO2"].diff().fillna(0)
    df["SPO2_TREND"] = df["SPO2_DIFF"].rolling(5, min_periods=1).mean()

    # Breathing features (if none available → zeros)
    df["BREATH_ANNOTATIONS"] = 0
    df["BREATH_COUNT"] = 0
    df["BREATH_RATE_AVG"] = 0

    # ---------------------------------
    # Ensure model feature order
    # ---------------------------------
    required_features = [
        'BREATH_ANNOTATIONS', 'HR', 'SPO2', 'HR_MEAN', 'HR_STD', 'HR_DIFF',
        'SPO2_DIFF', 'BREATH_COUNT', 'HR_VAR', 'SPO2_TREND', 'BREATH_RATE_AVG'
    ]

    for feat in required_features:
        if feat not in df.columns:
            df[feat] = 0

    # ---------------------------------
    # Scale (expecting the scaler to match these numeric columns)
    # ---------------------------------
    df[required_features] = scaler.transform(df[required_features])

    X = df[required_features].values

    # ---------------------------------
    # Predict with the RF model
    # ---------------------------------
    disease_pred = disease_model.predict(X)

    risk_results = []
    recommendations_list = []

    for p in disease_pred:
        if p == 0:
            risk_results.append("Low Risk")
            recommendations_list.append([
                "Maintain regular physical activity",
                "Eat a balanced diet rich in fruits and vegetables",
                "Ensure adequate sleep and hydration",
                "Continue regular health monitoring"
            ])
        else:
            risk_results.append("High Risk")
            recommendations_list.append([
                "Consult a healthcare professional promptly",
                "Increase daily physical activity and reduce sedentary behavior",
                "Adopt a heart-healthy diet",
                "Monitor vital signs closely",
                "Consider preventive screenings"
            ])

    return {
        "disease_risk_prediction": risk_results,
        "lifestyle_recommendations": recommendations_list
    }


# -------------------------------
# Anomaly Detection Helper
# -------------------------------
def preprocess_and_predict(df):
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0
    if 'heart_rate' in df.columns:
        df['HR_DIFF'] = df['heart_rate'].diff().fillna(0)
        df['HR_ANOMALY'] = df['heart_rate'].apply(lambda x: 1 if (x < 50 or x > 100) else 0)
    if 'SPO2' in df.columns:
        df['SPO2_DIFF'] = df['SPO2'].diff().fillna(0)
        df['SPO2_ANOMALY'] = df['SPO2'].apply(lambda x: 1 if x < 90 else 0)
    if 'BREATH' in df.columns:
        df['BREATH_COUNT'] = df['BREATH'].rolling(window=60, min_periods=1).sum()
    if 'ACTIVITY_ENERGY' in df.columns:
        df['ACTIVITY_ENERGY'] = df['ACTIVITY_ENERGY'].rolling(window=30, min_periods=1).apply(lambda x: np.sum(x**2))
    df[numeric_features] = scaler.transform(df[numeric_features])
    X = df[features].values
    rf_pred = rf_model.predict(X)
    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred == -1, 1, 0)
    rf_result = ["Normal" if p == 0 else "Anomaly" for p in rf_pred]
    iso_result = ["Normal" if p == 0 else "Anomaly" for p in iso_pred]
    return {
        "rf_predictions": rf_result,
        "iso_predictions": iso_result
    }

# -------------------------------
# Run Flask App
# -------------------------------
if __name__ == '__main__':
    app.run(debug=True)
