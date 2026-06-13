"""
generate_data.py
Generates synthetic sensor data for 5 equipment types with realistic
degradation patterns leading to failure events.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

EQUIPMENT = [
    {"id": "EQ-001", "name": "Compressor A", "type": "compressor"},
    {"id": "EQ-002", "name": "Pump B",        "type": "pump"},
    {"id": "EQ-003", "name": "Motor C",        "type": "motor"},
    {"id": "EQ-004", "name": "Conveyor D",     "type": "conveyor"},
    {"id": "EQ-005", "name": "Turbine E",      "type": "turbine"},
]

# Per-equipment sensor configs: (nominal_mean, nominal_std, failure_delta, unit)
SENSOR_CONFIGS = {
    "compressor": {
        "temperature": (65.0, 2.0, +30.0, "°C"),
        "vibration":   (2.0,  0.3, +7.0,  "mm/s"),
        "pressure":    (120.0,3.0, +40.0, "bar"),
        "current":     (35.0, 1.5, +15.0, "A"),
    },
    "pump": {
        "temperature": (55.0, 1.5, +25.0, "°C"),
        "vibration":   (1.8,  0.2, +6.0,  "mm/s"),
        "flow_rate":   (220.0,5.0, -60.0, "L/min"),
        "pressure":    (80.0, 2.0, +30.0, "bar"),
    },
    "motor": {
        "temperature": (50.0, 1.5, +28.0, "°C"),
        "vibration":   (1.5,  0.2, +6.5,  "mm/s"),
        "current":     (30.0, 1.0, +20.0, "A"),
        "rpm":         (2980.0,10.0,-300.0,"RPM"),
    },
    "conveyor": {
        "temperature": (60.0, 2.0, +22.0, "°C"),
        "vibration":   (2.5,  0.3, +5.0,  "mm/s"),
        "belt_tension":(850.0,15.0,-250.0,"N"),
        "speed":       (1.8,  0.05,-0.6,  "m/s"),
    },
    "turbine": {
        "temperature": (45.0, 1.5, +35.0, "°C"),
        "vibration":   (1.0,  0.15,+7.5,  "mm/s"),
        "rpm":         (1490.0,8.0,-200.0,"RPM"),
        "efficiency":  (94.0, 0.5, -12.0, "%"),
    },
}

def generate_equipment_data(equip, days=365, samples_per_day=24):
    """
    Generate time-series sensor data for one piece of equipment.
    Inserts 3-5 failure events, each preceded by a gradual degradation window
    of 7-21 days. Between failures, sensors return to nominal + slight wear offset.
    Returns a DataFrame with columns: timestamp, equipment_id, sensor_*, failure_in_Xd, label
    """
    eq_type   = equip["type"]
    eq_id     = equip["id"]
    sensors   = SENSOR_CONFIGS[eq_type]
    n_samples = days * samples_per_day

    start = datetime(2024, 1, 1)
    timestamps = [start + timedelta(hours=i) for i in range(n_samples)]

    # Schedule 3-5 failure events at random points (not too early, not too close)
    n_failures = np.random.randint(3, 6)
    failure_indices = sorted(np.random.choice(
        range(int(n_samples * 0.15), int(n_samples * 0.95)),
        size=n_failures, replace=False
    ))

    # Degradation window in samples (7-21 days * 24 samples/day)
    deg_windows = [np.random.randint(7 * 24, 21 * 24) for _ in failure_indices]

    # Build per-sensor arrays
    sensor_data = {s: np.zeros(n_samples) for s in sensors}
    labels      = np.zeros(n_samples, dtype=int)   # 0=normal, 1=failure
    rul_days    = np.full(n_samples, 999.0)         # remaining useful life in days

    # Cumulative wear offset (equipment degrades slightly over its life)
    wear_factor = 0.0

    for i in range(n_samples):
        wear_factor = min(i / n_samples * 0.15, 0.15)  # max 15% drift over lifetime

        # Determine if we're inside a degradation window
        deg_frac = 0.0   # 0 = nominal, 1 = at failure point
        min_rul  = 999.0

        for fi, (fail_idx, deg_win) in enumerate(zip(failure_indices, deg_windows)):
            deg_start = fail_idx - deg_win
            if deg_start <= i <= fail_idx:
                frac = (i - deg_start) / deg_win
                # Exponential degradation curve: slow at first, accelerates near failure
                deg_frac = max(deg_frac, frac ** 1.8)
                rul = (fail_idx - i) / samples_per_day
                min_rul = min(min_rul, rul)
            if i == fail_idx:
                labels[i] = 1

        rul_days[i] = min_rul

        for sensor_name, (nom_mean, nom_std, fail_delta, _) in sensors.items():
            # Apply wear offset
            worn_mean = nom_mean + nom_mean * wear_factor * np.sign(fail_delta) * 0.3
            # Interpolate toward failure value
            target = worn_mean + fail_delta * deg_frac
            noise  = np.random.normal(0, nom_std * (1 + deg_frac * 0.5))
            sensor_data[sensor_name][i] = target + noise

    rows = []
    for i in range(n_samples):
        row = {
            "timestamp":    timestamps[i].isoformat(),
            "equipment_id": eq_id,
            "equipment_type": eq_type,
            "label":        labels[i],
            "rul_days":     round(min(rul_days[i], 365.0), 2),
        }
        for s, arr in sensor_data.items():
            row[f"sensor_{s}"] = round(arr[i], 3)
        rows.append(row)

    return pd.DataFrame(rows)


def main():
    all_dfs = []
    for eq in EQUIPMENT:
        print(f"  Generating data for {eq['id']} ({eq['type']})...")
        df = generate_equipment_data(eq, days=365, samples_per_day=24)
        all_dfs.append(df)
        print(f"    → {len(df)} rows, {df['label'].sum()} failure events")

    combined = pd.concat(all_dfs, ignore_index=True)
    out_path = os.path.join(os.path.dirname(__file__), "../data/sensor_data.csv")
    combined.to_csv(out_path, index=False)
    print(f"\nSaved {len(combined)} total rows to {out_path}")
    print(f"Failure rate: {combined['label'].mean()*100:.2f}%")
    return combined


if __name__ == "__main__":
    main()
