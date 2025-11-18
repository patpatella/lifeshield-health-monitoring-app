# simulator/live_dashboard.py
import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from datetime import datetime
import time

API_URL = "http://127.0.0.1:5000/predict_batch"  # Flask API endpoint
BATCH_SIZE = 5
INTERVAL = 2  # seconds between batches

activity_levels = ['low', 'moderate', 'high']

# Data storage
timestamps = []
heart_rates = []
blood_oxygens = []
predictions = []

def generate_health_data(batch_size=BATCH_SIZE):
    data = []
    now = datetime.now()
    for i in range(batch_size):
        record_time = now + pd.Timedelta(seconds=i)
        record = {
            "timestamp": record_time.strftime("%Y-%m-%d %H:%M:%S"),
            "heart_rate": np.random.randint(60, 100),
            "blood_oxygen": np.random.randint(90, 100),
            "activity_level": np.random.choice(activity_levels)
        }
        data.append(record)
    return data

def send_batch(data):
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        print("Error:", response.status_code, response.text)
        return None

# Set up Matplotlib figure
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
plt.subplots_adjust(hspace=0.5)

ax1.set_title("Heart Rate with Anomalies")
ax1.set_xlabel("Timestamp")
ax1.set_ylabel("Heart Rate (bpm)")
ax2.set_title("Blood Oxygen Level")
ax2.set_xlabel("Timestamp")
ax2.set_ylabel("SpO2 (%)")

def update(frame):
    global timestamps, heart_rates, blood_oxygens, predictions
    # Generate new batch and get predictions
    batch_data = generate_health_data()
    result = send_batch(batch_data)
    if result:
        timestamps.extend(result["timestamps"])
        preds = result["predictions"]
        heart_rates.extend([d["heart_rate"] for d in batch_data])
        blood_oxygens.extend([d["blood_oxygen"] for d in batch_data])
        predictions.extend(preds)

    # Limit displayed points
    max_points = 50
    ts = timestamps[-max_points:]
    hr = heart_rates[-max_points:]
    spo2 = blood_oxygens[-max_points:]
    pred = predictions[-max_points:]

    # Clear previous plots
    ax1.cla()
    ax2.cla()

    # Plot heart rate
    colors = ['red' if p=="Anomaly" else 'green' for p in pred]
    ax1.scatter(ts, hr, c=colors)
    ax1.plot(ts, hr, color='blue', alpha=0.3)
    ax1.set_title("Heart Rate with Anomalies")
    ax1.set_xlabel("Timestamp")
    ax1.set_ylabel("Heart Rate (bpm)")
    ax1.tick_params(axis='x', rotation=45)

    # Plot blood oxygen
    ax2.plot(ts, spo2, marker='o', color='purple')
    ax2.set_title("Blood Oxygen Level")
    ax2.set_xlabel("Timestamp")
    ax2.set_ylabel("SpO2 (%)")
    ax2.tick_params(axis='x', rotation=45)

ani = FuncAnimation(fig, update, interval=INTERVAL*1000)
plt.show()
