# simulator/smartwatch_simulator.py
import requests
import pandas as pd
import numpy as np
import time
from datetime import datetime

API_URL = "http://127.0.0.1:5000/predict_batch"  # Flask API endpoint
BATCH_SIZE = 5  # number of records sent per batch
INTERVAL = 2  # seconds between batches

# Activity levels
activity_levels = ['low', 'moderate', 'high']

def generate_health_data(batch_size=BATCH_SIZE):
    """
    Simulate wearable health data
    """
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
    """
    Send batch data to Flask API
    """
    response = requests.post(API_URL, json=data)
    if response.status_code == 200:
        result = response.json()
        print("Timestamps:", result["timestamps"])
        print("Predictions:", result["predictions"])
    else:
        print("Error:", response.status_code, response.text)

if __name__ == "__main__":
    print("Starting smartwatch simulator...")
    try:
        while True:
            batch_data = generate_health_data()
            send_batch(batch_data)
            time.sleep(INTERVAL)
    except KeyboardInterrupt:
        print("Simulation stopped by user.")
