"""
inference.py
Loads trained model artifacts and exposes predict() for a given equipment type
and current sensor readings. Called by the FastAPI server.
"""

import os
import json
import numpy as np
import joblib
from collections import deque
from typing import Dict, Optional

MODELS_DIR = os.path.join(os.path.dirname(__file__), "../models")

WINDOW_SIZES = [3, 6, 12]
MAX_HISTORY  = max(WINDOW_SIZES)


_history: Dict[str, deque] = {}


def _load_artifact(eq_type: str, filename: str):
    path = os.path.join(MODELS_DIR, eq_type, filename)
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def _load_meta(eq_type: str) -> Optional[dict]:
    path = os.path.join(MODELS_DIR, eq_type, "meta.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _engineer_features(history_arr: np.ndarray, sensor_cols: list) -> np.ndarray:
    """
    Given a 2D array (n_history, n_sensors), compute rolling mean/std features
    for the LAST row using the full history buffer.
    Returns a 1D feature vector matching training feature_cols order.
    """
    features = list(history_arr[-1])  
    for col_idx in range(len(sensor_cols)):
        col_vals = history_arr[:, col_idx]
        for w in WINDOW_SIZES:
            window = col_vals[-w:]
            features.append(float(np.mean(window)))
            features.append(float(np.std(window)) if len(window) > 1 else 0.0)
    return np.array(features, dtype=np.float32)


def predict(equipment_id: str, eq_type: str, sensor_values: Dict[str, float]) -> dict:
    """
    Run all three models for one equipment reading.
    Returns:
      anomaly_score    : float 0-1 (higher = more anomalous)
      failure_prob     : float 0-1 (probability of imminent failure)
      predicted_rul    : float days (remaining useful life, None if not in degradation zone)
      health_score     : float 0-100
      alert_level      : "critical" | "warning" | "normal"
      recommendations  : list[str]
    """
    meta = _load_meta(eq_type)
    if meta is None:
        return {"error": f"No trained model found for equipment type '{eq_type}'"}

    sensor_cols = meta["sensor_cols"]

    raw = np.array([sensor_values.get(col.replace("sensor_", ""), 0.0) for col in sensor_cols],
                   dtype=np.float32)

   
    key = equipment_id
    if key not in _history:
        _history[key] = deque([raw] * MAX_HISTORY, maxlen=MAX_HISTORY)
    _history[key].append(raw)
    history_arr = np.array(list(_history[key]))

    feature_vec = _engineer_features(history_arr, sensor_cols)
    X = feature_vec.reshape(1, -1)


    iso        = _load_artifact(eq_type, "isolation_forest.pkl")
    scaler_iso = _load_artifact(eq_type, "scaler_iso.pkl")
    anomaly_score = 0.5
    if iso and scaler_iso:
        X_scaled = scaler_iso.transform(X)
        raw_score = iso.decision_function(X_scaled)[0]
        
        anomaly_score = float(np.clip(0.5 - raw_score, 0, 1))

    
    clf        = _load_artifact(eq_type, "classifier.pkl")
    scaler_cls = _load_artifact(eq_type, "scaler_cls.pkl")
    failure_prob = anomaly_score * 0.3  
    if clf and scaler_cls:
        X_scaled     = scaler_cls.transform(X)
        failure_prob = float(clf.predict_proba(X_scaled)[0][1])

    
    reg        = _load_artifact(eq_type, "rul_regressor.pkl")
    scaler_rul = _load_artifact(eq_type, "scaler_rul.pkl")
    predicted_rul = None
    if reg and scaler_rul and failure_prob > 0.25:
        X_scaled      = scaler_rul.transform(X)
        predicted_rul = float(np.clip(reg.predict(X_scaled)[0], 0, 90))

    
    health_score = round(max(0.0, min(100.0, (1 - failure_prob) * 80 + (1 - anomaly_score) * 20)), 1)

    
    if failure_prob >= 0.60 or (predicted_rul is not None and predicted_rul < 7):
        alert_level = "critical"
    elif failure_prob >= 0.30 or anomaly_score >= 0.55:
        alert_level = "warning"
    else:
        alert_level = "normal"

    
    recs = _generate_recommendations(eq_type, alert_level, failure_prob, predicted_rul, sensor_values)

    return {
        "equipment_id":   equipment_id,
        "equipment_type": eq_type,
        "anomaly_score":  round(anomaly_score, 4),
        "failure_prob":   round(failure_prob, 4),
        "failure_prob_pct": round(failure_prob * 100, 1),
        "predicted_rul":  round(predicted_rul, 1) if predicted_rul is not None else None,
        "health_score":   health_score,
        "alert_level":    alert_level,
        "recommendations": recs,
    }


def _generate_recommendations(eq_type, alert_level, failure_prob, rul, sensors) -> list:
    recs = []
    fp_pct = failure_prob * 100

    if alert_level == "critical":
        recs.append(f"IMMEDIATE: Schedule emergency inspection within 24 hours (failure probability {fp_pct:.0f}%)")
        if rul is not None:
            recs.append(f"Estimated {rul:.0f} days until failure — reduce operational load immediately")
    elif alert_level == "warning":
        recs.append(f"Schedule maintenance within 7 days (failure probability {fp_pct:.0f}%)")
    else:
        recs.append("Continue normal operation — run scheduled maintenance per calendar")

    
    THRESHOLDS = {
        "compressor": {"temperature": 85, "vibration": 6.0, "pressure": 145, "current": 48},
        "pump":       {"temperature": 70, "vibration": 5.0, "flow_rate": 185, "pressure": 100},
        "motor":      {"temperature": 68, "vibration": 5.5, "current": 45, "rpm": 3050},
        "conveyor":   {"temperature": 68, "vibration": 5.5, "belt_tension": 700, "speed": 1.65},
        "turbine":    {"temperature": 65, "vibration": 4.5, "rpm": 1550, "efficiency": 88},
    }
    thresholds = THRESHOLDS.get(eq_type, {})
    for sensor, val in sensors.items():
        thresh = thresholds.get(sensor)
        if thresh is None:
            continue
        
        if sensor not in ("flow_rate", "belt_tension", "speed", "rpm", "efficiency"):
            if val > thresh:
                recs.append(f"Inspect {sensor.replace('_',' ')}: {val:.1f} exceeds warning threshold ({thresh})")
        else:
            
            if val < thresh:
                recs.append(f"Check {sensor.replace('_',' ')}: {val:.1f} below safe threshold ({thresh})")

    return recs
