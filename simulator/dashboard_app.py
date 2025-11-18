import tkinter as tk
from tkinter import ttk
import requests
import pandas as pd
import numpy as np
from datetime import datetime
import threading
import time
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

API_URL = "http://127.0.0.1:5000/predict_batch"
BATCH_SIZE = 5
INTERVAL = 2  # seconds
activity_levels = ['low', 'moderate', 'high']
activity_mapping = {'low': 1, 'moderate': 2, 'high': 3}

timestamps, heart_rates, blood_oxygens, activity_vals, predictions = [], [], [], [], []
running = False

# Tkinter root
root = tk.Tk()
root.title("Real-Time Health Dashboard")
root.geometry("1200x800")
default_bg = root.cget("bg")

# Flash GUI on anomaly
def flash_alert():
    root.config(bg="red")
    root.after(300, lambda: root.config(bg=default_bg))

def generate_health_data(batch_size=BATCH_SIZE):
    now = datetime.now()
    data = []
    for i in range(batch_size):
        record_time = now + pd.Timedelta(seconds=i)
        data.append({
            "timestamp": record_time.strftime("%Y-%m-%d %H:%M:%S"),
            "heart_rate": np.random.randint(60, 100),
            "blood_oxygen": np.random.randint(90, 100),
            "activity_level": np.random.choice(activity_levels)
        })
    return data

def send_batch(data):
    try:
        response = requests.post(API_URL, json=data)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print("API Error:", e)
    return None

def update_dashboard():
    if timestamps:
        latest_hr = heart_rates[-1]
        latest_spo2 = blood_oxygens[-1]
        latest_act = activity_vals[-1]
        latest_pred = predictions[-1]

        avg_hr = np.mean(heart_rates[-10:]) if heart_rates else 0
        avg_spo2 = np.mean(blood_oxygens[-10:]) if blood_oxygens else 0
        anomaly_pct = predictions.count("Anomaly") / max(len(predictions), 1) * 100

        # Update labels
        latest_hr_var.set(f"💓 Heart Rate: {latest_hr} bpm")
        latest_spo2_var.set(f"🩸 SpO₂: {latest_spo2}%")
        latest_act_var.set(f"🏃 Activity: {latest_act}")
        latest_pred_var.set(f"🔮 Prediction: {latest_pred}")
        avg_hr_var.set(f"Avg HR (10): {avg_hr:.1f} bpm")
        avg_spo2_var.set(f"Avg SpO₂ (10): {avg_spo2:.1f}%")
        anomaly_var.set(f"⚠️ Anomalies: {anomaly_pct:.1f}%")

def update_plot():
    max_points = 50
    ts = timestamps[-max_points:]
    hr = heart_rates[-max_points:]
    spo2 = blood_oxygens[-max_points:]
    act = activity_vals[-max_points:]
    pred = predictions[-max_points:]

    x = list(range(len(ts)))
    x_spo2 = [i + 0.2 for i in x]
    x_act = [i + 0.4 for i in x]

    ax1.cla(); ax2.cla(); ax3.cla()

    # Heart rate
    colors = ['red' if p == "Anomaly" else 'green' for p in pred]
    ax1.scatter(x, hr, c=colors, s=40)
    ax1.plot(x, hr, alpha=0.3, linewidth=2)
    ax1.set_title("Heart Rate with Anomalies", fontsize=12, fontweight="bold")
    ax1.set_ylabel("HR (bpm)")
    ax1.grid(True, linestyle="--", alpha=0.3)

    # SpO2
    ax2.plot(x_spo2, spo2, marker='o', linewidth=2, color='purple')
    ax2.set_title("Blood Oxygen Level", fontsize=12, fontweight="bold")
    ax2.set_ylabel("SpO₂ (%)")
    ax2.grid(True, linestyle="--", alpha=0.3)

    # Activity level
    ax3.plot(x_act, act, marker='s', linewidth=2, color='orange')
    ax3.set_title("Activity Level", fontsize=12, fontweight="bold")
    ax3.set_ylabel("Level (1-3)")
    ax3.set_yticks([1, 2, 3])
    ax3.set_yticklabels(['low', 'moderate', 'high'])
    ax3.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    canvas.draw()

def run_simulation():
    global running
    while running:
        batch_data = generate_health_data()
        result = send_batch(batch_data)
        if result:
            timestamps.extend(result["timestamps"])
            preds = result["predictions"]
            heart_rates.extend([d["heart_rate"] for d in batch_data])
            blood_oxygens.extend([d["blood_oxygen"] for d in batch_data])
            activity_vals.extend([activity_mapping[d["activity_level"]] for d in batch_data])
            predictions.extend(preds)

            if "Anomaly" in preds:
                flash_alert()

        update_dashboard()
        update_plot()
        time.sleep(INTERVAL)

def start_sim():
    global running
    if not running:
        running = True
        threading.Thread(target=run_simulation, daemon=True).start()

def stop_sim():
    global running
    running = False

# --- Buttons at the very top ---
btn_frame = ttk.Frame(root)
btn_frame.pack(side=tk.TOP, fill=tk.X, pady=5)

ttk.Button(btn_frame, text="Start Simulation", command=start_sim).pack(side=tk.LEFT, padx=10)
ttk.Button(btn_frame, text="Stop Simulation", command=stop_sim).pack(side=tk.LEFT, padx=10)

# --- Main layout (plots + status panel) ---
main_frame = ttk.Frame(root)
main_frame.pack(fill=tk.BOTH, expand=True)

plot_frame = ttk.Frame(main_frame)
plot_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=10, pady=10)

status_frame = ttk.LabelFrame(main_frame, text="Live Metrics", padding=10)
status_frame.pack(side=tk.RIGHT, fill=tk.Y, padx=10, pady=10)

# --- Status variables and labels ---
latest_hr_var = tk.StringVar(value="💓 Heart Rate: -")
latest_spo2_var = tk.StringVar(value="🩸 SpO₂: -")
latest_act_var = tk.StringVar(value="🏃 Activity: -")
latest_pred_var = tk.StringVar(value="🔮 Prediction: -")
avg_hr_var = tk.StringVar(value="Avg HR (10): -")
avg_spo2_var = tk.StringVar(value="Avg SpO₂ (10): -")
anomaly_var = tk.StringVar(value="⚠️ Anomalies: -")

for var in [latest_hr_var, latest_spo2_var, latest_act_var, latest_pred_var, avg_hr_var, avg_spo2_var, anomaly_var]:
    ttk.Label(status_frame, textvariable=var, font=("Segoe UI", 11)).pack(anchor="w", pady=3)

# --- Matplotlib figure with 3 vertical subplots ---
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 8))
canvas = FigureCanvasTkAgg(fig, master=plot_frame)
canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

root.mainloop()
