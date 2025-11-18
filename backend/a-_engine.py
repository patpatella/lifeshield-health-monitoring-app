# backend/ai_engine.py
import os
import joblib
import numpy as np
import pandas as pd

# Paths (adjust if you will store disease models elsewhere)
BASE_MODEL_PATH = os.path.join("backend", "data", "bidmc", "models")
DISEASE_MODEL_PATH = os.path.join(BASE_MODEL_PATH, "disease_risk_model.joblib")
DISEASE_FEATURES_PATH = os.path.join(BASE_MODEL_PATH, "disease_features.pkl")

# A simple, human-readable mapping from risk labels to recommendations.
RECOMMENDATION_MAP = {
    "cardiovascular": {
        "low": [
            "Keep up regular moderate physical activity (150 min/week).",
            "Maintain a balanced diet rich in vegetables and whole grains.",
            "Monitor blood pressure regularly."
        ],
        "medium": [
            "Schedule a checkup with a cardiologist.",
            "Reduce sodium intake and avoid smoking.",
            "Begin a structured exercise plan; aim for at least 150 min moderate-intensity activity per week."
        ],
        "high": [
            "Seek urgent medical evaluation for cardiovascular risk.",
            "Stop smoking and avoid heavy exertion until cleared by a clinician.",
            "Follow-up for cholesterol and blood pressure management; consider referral to a specialist."
        ]
    },
    "respiratory": {
        "low": [
            "Avoid exposure to smoke and pollutants.",
            "Keep active and practice breathing exercises."
        ],
        "medium": [
            "Consider spirometry and consult primary care if symptoms persist.",
            "Monitor SpO₂ and note patterns (e.g., at rest vs activity)."
        ],
        "high": [
            "Seek urgent medical review: low SpO₂ or breathing distress requires immediate attention.",
            "If you have COPD/asthma history, follow your rescue plan and contact your clinician."
        ]
    },
    "sleep_apnea": {
        "low": ["Maintain healthy weight and sleep hygiene."],
        "medium": ["Discuss sleep symptoms with your clinician; consider home sleep test."],
        "high": ["Refer for sleep clinic/PSG study — high risk for obstructive sleep apnea."]
    },
    "general": {
        "low": ["Maintain current healthy habits."],
        "medium": ["Increase monitoring frequency and consider clinician review."],
        "high": ["Immediate clinical assessment recommended."]
    }
}

def load_disease_model():
    """Try to load a trained disease risk model; return (model, features) or (None, None)."""
    if os.path.exists(DISEASE_MODEL_PATH) and os.path.exists(DISEASE_FEATURES_PATH):
        model = joblib.load(DISEASE_MODEL_PATH)
        features = joblib.load(DISEASE_FEATURES_PATH)
        return model, features
    return None, None

def heuristic_risk_scores(df: pd.DataFrame):
    """
    Compute interpretable heuristic risk scores [0..1] for demo diseases based on features present.
    Input: df - rows of preprocessed signals/features (columns like: HR_MEAN, HR_STD, SPO2, BREATH_COUNT, ACTIVITY_ENERGY)
    Returns: list of dicts (one per row) with risk scores and categories.
    """
    results = []
    for _, row in df.iterrows():
        # Default values (guard for missing columns)
        hr = float(row.get("HR", row.get("HR_MEAN", np.nan)) or np.nan)
        spo2 = float(row.get("SPO2", np.nan) or np.nan)
        breath = float(row.get("BREATH_COUNT", np.nan) or np.nan)
        activity = float(row.get("ACTIVITY_ENERGY", np.nan) or 0)

        # Cardiovascular risk heuristic:
        # base on high HR, high HR variability, low activity (simple)
        hr_risk = 0.0
        if not np.isnan(hr):
            if hr < 50 or hr > 100:
                hr_risk = 0.8
            elif hr > 90:
                hr_risk = 0.5
            else:
                hr_risk = 0.1
        # adjust by activity (low activity -> slightly higher risk)
        if activity < 10:
            hr_risk = min(1.0, hr_risk + 0.15)

        # Respiratory risk heuristic:
        resp_risk = 0.0
        if not np.isnan(spo2):
            if spo2 < 88:
                resp_risk = 0.95
            elif spo2 < 92:
                resp_risk = 0.7
            elif spo2 < 95:
                resp_risk = 0.3
            else:
                resp_risk = 0.05
        # breathing count extremes increase risk
        if not np.isnan(breath):
            if breath < 8 or breath > 30:
                resp_risk = max(resp_risk, 0.7)

        # Sleep apnea proxy (high HR variability + high breath_count at night would raise this)
        sleep_apnea_risk = 0.0
        # use surrogate: high breath_count variability or frequent desaturations -> raise risk
        if not np.isnan(spo2) and spo2 < 92:
            sleep_apnea_risk = max(sleep_apnea_risk, 0.4)
        if 'HR_STD' in row.index and not np.isnan(row.get('HR_STD')):
            if row['HR_STD'] > 7:
                sleep_apnea_risk = max(sleep_apnea_risk, 0.4)

        # Ensure clipped to [0,1]
        cv = float(np.clip(hr_risk, 0.0, 1.0))
        resp = float(np.clip(resp_risk, 0.0, 1.0))
        sleep = float(np.clip(sleep_apnea_risk, 0.0, 1.0))

        results.append({
            "cardiovascular_score": cv,
            "respiratory_score": resp,
            "sleep_apnea_score": sleep
        })
    return results

def scores_to_level(score):
    """Map numeric score to low/medium/high label."""
    if score >= 0.75:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"

def generate_recommendations_from_scores(score_dict):
    """
    Convert a single score_dict (returned by heuristic or model) into prioritized recommendations.
    score_dict example: {"cardiovascular_score": 0.3, ...}
    """
    recs = {}
    for key, score in score_dict.items():
        topic = "general"
        if "cardio" in key:
            topic = "cardiovascular"
        elif "respir" in key:
            topic = "respiratory"
        elif "sleep" in key:
            topic = "sleep_apnea"

        level = scores_to_level(score)
        recs[key] = {
            "score": score,
            "level": level,
            "recommendations": RECOMMENDATION_MAP.get(topic, RECOMMENDATION_MAP['general']).get(level, [])
        }
    return recs

def predict_disease_risks(df: pd.DataFrame):
    """
    Attempt to run a trained disease model (if available) returning probabilities,
    otherwise fall back to heuristic_risk_scores().
    Returns list-of-dict (one per row) with risk scores and recommended actions.
    """
    model, model_features = load_disease_model()
    outputs = []
    if model is not None and model_features is not None:
        # Ensure columns exist and are in expected order; fill missing with 0
        X = pd.DataFrame(columns=model_features)
        for col in model_features:
            X[col] = df[col] if col in df.columns else 0.0

        # If scaler was used during disease training you should load/transform similarly.
        # For now assume model expects raw features as saved
        probs = None
        try:
            probs = model.predict_proba(X)
            # assume model.classes_ contains disease labels or [0,1] - handle carefully in your training
            # Here, we assume binary per disease; this is placeholder logic
            for p in probs:
                # Example: if model returns 2-class prob [p_normal, p_disease]
                disease_prob = float(p[-1]) if len(p) > 1 else float(p[0])
                outputs.append({"model_based_score": disease_prob})
        except Exception:
            # fallback to heuristic if model can't produce probabilities
            outputs = heuristic_risk_scores(df)
    else:
        outputs = heuristic_risk_scores(df)

    # For each output produce recommendation mapping
    results_with_recs = []
    for out in outputs:
        recs = generate_recommendations_from_scores(out)
        results_with_recs.append({
            "scores": out,
            "recommendations": recs
        })
    return results_with_recs
