import os
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# -------------------------------
# Model Paths (UPDATED)
# -------------------------------
BASE_MODEL_PATH = 'models'
ISO_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_isolation_forest.joblib')
RF_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_random_forest.joblib')
SCALER_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_scaler.joblib')
FEATURES_PATH = os.path.join(BASE_MODEL_PATH, 'physionet_features.pkl')
DISEASE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, 'disease_rf_model.joblib')

# -------------------------------
# Load Trained Objects
# -------------------------------
iso_model = joblib.load(ISO_MODEL_PATH)
rf_model = joblib.load(RF_MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
features = joblib.load(FEATURES_PATH)
disease_model = joblib.load(DISEASE_MODEL_PATH)

numeric_features = features

# -------------------------------
# Logging Setup
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
CORS(app)   # <---- FIXED CORS

@app.route('/')
def index():
    return "AI Health Monitoring API is running!"

# -------------------------------
# Helper
# -------------------------------
def convert_numpy(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

# -------------------------------
# MAIN PREDICTION ENDPOINT
# -------------------------------
@app.route('/predict_all', methods=['POST'])
def predict_all():
    data = request.get_json()

    if data is None:
        return jsonify({"error": "No input data provided"}), 400

    if isinstance(data, dict):
        data = [data]
    elif not isinstance(data, list):
        return jsonify({"error": "Input must be dict or list"}), 400

    df = pd.DataFrame(data)

    anomaly_results = preprocess_and_predict(df)
    disease_results = preprocess_disease(df)

    combined_results = []
    logs = []
    now = datetime.now().isoformat()

    for i in range(len(df)):
        record = {
            "record_index": int(i),
            "timestamp": str(df['timestamp'].iloc[i]) if 'timestamp' in df else now,
            "heart_rate": int(df['heart_rate'].iloc[i]) if 'heart_rate' in df else 0,
            "SPO2": int(df['SPO2'].iloc[i]) if 'SPO2' in df else 0,
            "activity_level": str(df['activity_level'].iloc[i]) if 'activity_level' in df else "moderate",
            "anomaly_rf": str(anomaly_results['rf_predictions'][i]),
            "anomaly_iso": str(anomaly_results['iso_predictions'][i]),
            "disease_risk": str(disease_results['disease_risk_prediction'][i]),
            "lifestyle_recommendations": [str(r) for r in disease_results['lifestyle_recommendations'][i]]
        }

        record = {k: convert_numpy(v) for k, v in record.items()}
        combined_results.append(record)

        logs.append({
            "timestamp": record["timestamp"],
            "record_index": record["record_index"],
            "rf_anomaly": record["anomaly_rf"],
            "iso_anomaly": record["anomaly_iso"],
            "disease_risk": record["disease_risk"],
            "lifestyle_recommendations": "|".join(record["lifestyle_recommendations"]),
            "activity_level": record["activity_level"]
        })

    pd.DataFrame(logs).to_csv(LOG_FILE, mode='a', header=False, index=False)

    return jsonify({"results": combined_results})

# -------------------------------
# GET LOGS
# -------------------------------
@app.route('/get_logs', methods=['GET'])
def get_logs():
    if not os.path.exists(LOG_FILE):
        return jsonify({"error": "No logs found"}), 404

    output_format = request.args.get('format', 'json')

    if output_format == "csv":
        return send_file(LOG_FILE, mimetype="text/csv", as_attachment=True,
                         download_name="health_predictions_log.csv")
    else:
        df = pd.read_csv(LOG_FILE)
        return jsonify(df.to_dict(orient='records'))

# -------------------------------
# DISEASE PREDICTION
# -------------------------------
def preprocess_disease(df):
    df = df.copy()
    df["HR"] = df.get("heart_rate", pd.Series([0]*len(df)))
    df["SPO2"] = df.get("SPO2", pd.Series([0]*len(df)))

    df["HR_MEAN"] = df["HR"].rolling(5, min_periods=1).mean()
    df["HR_STD"] = df["HR"].rolling(5, min_periods=1).std().fillna(0)
    df["HR_DIFF"] = df["HR"].diff().fillna(0)
    df["HR_VAR"] = df["HR"].rolling(5, min_periods=1).var().fillna(0)

    df["SPO2_DIFF"] = df["SPO2"].diff().fillna(0)
    df["SPO2_TREND"] = df["SPO2_DIFF"].rolling(5, min_periods=1).mean()

    df["BREATH_ANNOTATIONS"] = 0
    df["BREATH_COUNT"] = 0
    df["BREATH_RATE_AVG"] = 0

    for feat in features:
        if feat not in df.columns:
            df[feat] = 0

    df[features] = scaler.transform(df[features])
    preds = disease_model.predict(df[features])

    risks, recs = [], []
    for p in preds:
        if p == 0:
            risks.append("Low Risk")
            recs.append([
                "Maintain regular physical activity",
                "Eat a balanced diet",
                "Stay hydrated and sleep well",
                "Continue regular monitoring"
            ])
        else:
            risks.append("High Risk")
            recs.append([
                "Consult a healthcare professional",
                "Increase daily physical activity",
                "Adopt a heart-healthy diet",
                "Monitor vital signs closely",
                "Consider preventive screenings"
            ])
    return {"disease_risk_prediction": risks, "lifestyle_recommendations": recs}

# -------------------------------
# ANOMALY DETECTION
# -------------------------------
def preprocess_and_predict(df):
    df = df.copy()
    for feat in features:
        if feat not in df.columns:
            df[feat] = 0

    df[features] = scaler.transform(df[features])
    X = df[features].values

    rf_pred = rf_model.predict(X)
    iso_pred = iso_model.predict(X)
    iso_pred = np.where(iso_pred == -1, 1, 0)

    rf_result = ["Normal" if p == 0 else "Anomaly" for p in rf_pred]
    iso_result = ["Normal" if p == 0 else "Anomaly" for p in iso_pred]

    return {"rf_predictions": rf_result, "iso_predictions": iso_result}

# -------------------------------
# Flask Run
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
