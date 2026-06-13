"""
train_models.py
Trains three models per equipment type:
  1. IsolationForest  — unsupervised anomaly detection (no labels needed)
  2. RandomForest     — binary failure classification (label=0/1)
  3. GradientBoosting — RUL regression (days until next failure)

Saves all artifacts to models/<equipment_type>/
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest, RandomForestClassifier, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, roc_auc_score,
    mean_absolute_error, r2_score
)

DATA_PATH   = os.path.join(os.path.dirname(__file__), "../data/sensor_data.csv")
MODELS_DIR  = os.path.join(os.path.dirname(__file__), "../models")


WINDOW_SIZES = [3, 6, 12]

SENSOR_COLS_BY_TYPE = {
    "compressor": ["sensor_temperature", "sensor_vibration", "sensor_pressure", "sensor_current"],
    "pump":       ["sensor_temperature", "sensor_vibration", "sensor_flow_rate", "sensor_pressure"],
    "motor":      ["sensor_temperature", "sensor_vibration", "sensor_current", "sensor_rpm"],
    "conveyor":   ["sensor_temperature", "sensor_vibration", "sensor_belt_tension", "sensor_speed"],
    "turbine":    ["sensor_temperature", "sensor_vibration", "sensor_rpm", "sensor_efficiency"],
}


def engineer_features(df, sensor_cols):
    """Add rolling mean/std features per sensor column."""
    df = df.copy().reset_index(drop=True)
    feature_cols = list(sensor_cols)
    for col in sensor_cols:
        for w in WINDOW_SIZES:
            mean_col = f"{col}_mean{w}"
            std_col  = f"{col}_std{w}"
            df[mean_col] = df[col].rolling(w, min_periods=1).mean()
            df[std_col]  = df[col].rolling(w, min_periods=1).std().fillna(0)
            feature_cols += [mean_col, std_col]
    return df, feature_cols


def train_equipment_models(eq_type, df_eq):
    sensor_cols = SENSOR_COLS_BY_TYPE[eq_type]
    out_dir = os.path.join(MODELS_DIR, eq_type)
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'='*50}")
    print(f"Training models for: {eq_type.upper()}")
    print(f"  Samples: {len(df_eq)}  |  Failures: {df_eq['label'].sum()}")

    df_feat, feature_cols = engineer_features(df_eq, sensor_cols)
    X = df_feat[feature_cols].values
    y_cls = df_feat["label"].values
    y_rul = df_feat["rul_days"].values.clip(0, 90)  

    
    print("\n  [1/3] Training IsolationForest (anomaly detection)...")
    scaler_iso = StandardScaler()
    X_normal   = X[y_cls == 0]
    X_scaled   = scaler_iso.fit_transform(X_normal)

    iso = IsolationForest(
        n_estimators=200,
        contamination=0.05,
        random_state=42,
        n_jobs=-1
    )
    iso.fit(X_scaled)
    scores = iso.decision_function(scaler_iso.transform(X))

    anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    print(f"    Mean anomaly score (failures): {anomaly_scores[y_cls==1].mean():.3f}")
    print(f"    Mean anomaly score (normal):   {anomaly_scores[y_cls==0].mean():.3f}")

    joblib.dump(iso, os.path.join(out_dir, "isolation_forest.pkl"))
    joblib.dump(scaler_iso, os.path.join(out_dir, "scaler_iso.pkl"))
    print("\n  [2/3] Training RandomForest (failure classification)...")
    scaler_cls = StandardScaler()
    X_cls_scaled = scaler_cls.fit_transform(X)
    class_weight = {0: 1, 1: int((y_cls == 0).sum() / max((y_cls == 1).sum(), 1))}
    X_train, X_test, y_train, y_test = train_test_split(
        X_cls_scaled, y_cls, test_size=0.2, random_state=42, stratify=y_cls
    )
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight=class_weight,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred  = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_proba)
    print(f"    AUC-ROC: {auc:.4f}")
    print(f"    {classification_report(y_test, y_pred, target_names=['Normal','Failure'], zero_division=0)}")
    joblib.dump(clf, os.path.join(out_dir, "classifier.pkl"))
    joblib.dump(scaler_cls, os.path.join(out_dir, "scaler_cls.pkl"))
    print("\n  [3/3] Training GradientBoosting (RUL regression)...")
       mask = y_rul < 90
    if mask.sum() < 50:
        print("    Not enough near-failure samples for RUL regression, skipping.")
        reg = None
        scaler_rul = None
    else:
        scaler_rul = StandardScaler()
        X_rul = scaler_rul.fit_transform(X[mask])
        y_rul_sub = y_rul[mask]
        Xr_train, Xr_test, yr_train, yr_test = train_test_split(
            X_rul, y_rul_sub, test_size=0.2, random_state=42
        )
        reg = GradientBoostingRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            random_state=42
        )
        reg.fit(Xr_train, yr_train)
        yr_pred = reg.predict(Xr_test).clip(0, 90)
        mae = mean_absolute_error(yr_test, yr_pred)
        r2  = r2_score(yr_test, yr_pred)
        print(f"    MAE: {mae:.2f} days  |  R²: {r2:.4f}")

        joblib.dump(reg, os.path.join(out_dir, "rul_regressor.pkl"))
        joblib.dump(scaler_rul, os.path.join(out_dir, "scaler_rul.pkl"))
    meta = {
        "equipment_type": eq_type,
        "sensor_cols": sensor_cols,
        "feature_cols": feature_cols,
        "window_sizes": WINDOW_SIZES,
        "n_training_samples": int(len(df_eq)),
        "n_failure_events": int(y_cls.sum()),
        "auc_roc": round(auc, 4),
        "rul_mae_days": round(mae, 2) if reg else None,
        "rul_r2": round(r2, 4) if reg else None,
    }
    with open(os.path.join(out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved all artifacts to models/{eq_type}/")
    return meta
def main():
    print("Loading sensor data...")
    df = pd.read_csv(DATA_PATH)
    print(f"  {len(df)} rows loaded")
    all_meta = {}
    for eq_type in SENSOR_COLS_BY_TYPE:
        df_eq = df[df["equipment_type"] == eq_type].copy()
        if len(df_eq) == 0:
            print(f"No data for {eq_type}, skipping.")
            continue
        meta = train_equipment_models(eq_type, df_eq)
        all_meta[eq_type] = meta
    summary_path = os.path.join(MODELS_DIR, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_meta, f, indent=2)
    print(f"\n{'='*50}")
    print("Training complete. Model summary:")
    for eq_type, meta in all_meta.items():
        print(f"  {eq_type:12s} | AUC: {meta['auc_roc']:.4f} | RUL MAE: {meta.get('rul_mae_days','N/A')} days")
    print(f"\nSummary saved to {summary_path}")

if __name__ == "__main__":
    main()
