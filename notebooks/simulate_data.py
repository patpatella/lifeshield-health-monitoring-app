# notebooks/simulate_data.py
import pandas as pd
import numpy as np
import os

def generate_simulated_data(n_minutes=1440, start='2025-11-01'):
    rng = pd.date_range(start=start, periods=n_minutes, freq='T')
    # base circadian heart rate pattern + noise
    base_hr = 70 + 10 * np.sin(np.linspace(0, 2 * np.pi, n_minutes))
    hr = (base_hr + np.random.normal(0, 4, n_minutes)).round().astype(int)
    spo2 = (97 + np.random.normal(0, 1, n_minutes)).round(1)
    activity = np.random.choice(['low', 'moderate', 'high'], size=n_minutes, p=[0.6, 0.3, 0.1])
    df = pd.DataFrame({'timestamp': rng, 'heart_rate': hr, 'blood_oxygen': spo2, 'activity_level': activity})
    return df

def save_csv(df, out_path='../data/simulated.csv'):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved simulated data to {out_path}")

if __name__ == "__main__":
    df = generate_simulated_data(n_minutes=1440)  # 1 day of minute-level data
    save_csv(df, out_path='data/simulated.csv')
    print(df.head())
