# AI Health Monitor – Real-Time Anomaly Detection & Risk Prediction (SDG 3)

A full-stack AI-powered health monitoring system designed to support SDG 3 – Good Health & Well-Being by providing real-time detection of anomalies and potential cardiovascular risks using wearable-like data.

The system uses machine learning, cloud deployment, and a React web interface to provide accessible, scalable health insights to users.

# 🚀 Features
* Backend (Python + Flask + Cloud Run)

* Predicts anomalies using:

* Isolation Forest

* Random Forest Classifier

* Predicts disease risk using a separate Random Forest model.

* Generates personalized lifestyle recommendations.

* Logs timestamps & activity levels.

* Exposed /predict_all API endpoint.

* Fully containerized with Docker & deployed to Google Cloud Run.

* Frontend (React + Firebase Hosting)

* Clean UI for uploading/entering health data.

* Calls Cloud Run backend using secure fetch-based API calls.

# Displays:

* Predictions

* Risk status

* Recommendations

* Activity levels

* Timestamped results

# 🧠 Machine Learning Models

Trained using a mixture of real biosignal data and synthetic wearable-like samples.

# Models Included

* Isolation Forest – unsupervised anomaly detection

* Random Forest Anomaly Model – supervised anomaly detection

* Random Forest Risk Model – cardiovascular risk prediction


# 🔧 Tech Stack

# Machine Learning

* Python

* scikit-learn

* pandas, numpy

# Backend

* Flask

* Gunicorn

* Google Cloud Run

* Docker

# Frontend

* React

* Axios

* Firebase Hosting

# 🌍 SDG Alignment (SDG 3 – Good Health & Well-Being)

This project contributes to global health by enabling:

* Early detection of cardiovascular risks

* Real-time anomaly alerts

* Personalized lifestyle recommendations

* Low-cost & scalable access via the cloud

# 🛠️ Installation & Setup
Backend Setup
cd backend
pip install -r requirements.txt
python app.py

Run Backend Locally
export PORT=8080
python app.py

Build & Deploy (Cloud Run)
gcloud builds submit --tag gcr.io/PROJECT-ID/ai-health-monitor
gcloud run deploy ai-health-monitor \
  --image gcr.io/PROJECT-ID/ai-health-monitor \
  --platform managed --region us-central1 --allow-unauthenticated

# 🖥️ Frontend Setup
cd frontend
npm install
npm start

Build for Production
npm run build

Deploy to Firebase
firebase init
firebase deploy

🔌 API Endpoint
POST /predict_all

Example request:

[
  {
    "heart_rate": 75,
    "SPO2": 97,
    "activity_level": "moderate",
    "HR_MEAN": 74.5,
    "HR_STD": 1.2,
    "HR_DIFF": 0.5,
    "HR_VAR": 1.4,
    "SPO2_DIFF": 0.3,
    "SPO2_TREND": 0.2,
    "BREATH_ANNOTATIONS": 0,
    "BREATH_COUNT": 0,
    "BREATH_RATE_AVG": 0
  }
]

📊 Output Example
{
  "results": [
    {
      "record_index": 0,
      "heart_rate": 75,
      "SPO2": 97,
      "activity_level": "moderate",
      "anomaly_iso": "Normal",
      "anomaly_rf": "Normal",
      "disease_risk": "Low Risk",
      "lifestyle_recommendations": [
        "Maintain regular physical activity",
        "Eat a balanced diet",
        "Stay hydrated and sleep well",
        "Continue regular monitoring"
      ],
      "timestamp": "2025-11-19T14:53:07"
    }
  ]
}

# 🧪 Testing

* Manual batch testing using PowerShell or curl.

* Frontend integration tests with sample health metrics.

* CORS validation between Firebase → Cloud Run.

# 🧭 Future Enhancements

* Deep learning models (LSTM for ECG/PPG signals).

* Integration with actual wearable devices.

* User accounts and cloud-based health history.

* Continuous monitoring with IoT streaming.

* Doctor-facing dashboard for clinical review.