"""
server.py
FastAPI REST + WebSocket server for predictive maintenance.

Endpoints:
  GET  /health                          — server health check
  GET  /equipment                       — list all equipment
  POST /predict/{equipment_id}          — run inference on posted sensor data
  GET  /status/{equipment_id}           — latest cached prediction for an equipment
  GET  /alerts                          — all active alerts across fleet
  WS   /ws/live                         — WebSocket: streams live simulated readings every 2s
"""

import asyncio
import json
import random
import math
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from inference import predict

app = FastAPI(title="Predictive Maintenance API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

EQUIPMENT = [
    {"id": "EQ-001", "name": "Compressor A", "type": "compressor"},
    {"id": "EQ-002", "name": "Pump B",        "type": "pump"},
    {"id": "EQ-003", "name": "Motor C",        "type": "motor"},
    {"id": "EQ-004", "name": "Conveyor D",     "type": "conveyor"},
    {"id": "EQ-005", "name": "Turbine E",      "type": "turbine"},
]

NOMINAL_SENSORS = {
    "compressor": {"temperature": 65.0, "vibration": 2.0, "pressure": 120.0, "current": 35.0},
    "pump":       {"temperature": 55.0, "vibration": 1.8, "flow_rate": 220.0, "pressure": 80.0},
    "motor":      {"temperature": 50.0, "vibration": 1.5, "current": 30.0, "rpm": 2980.0},
    "conveyor":   {"temperature": 60.0, "vibration": 2.5, "belt_tension": 850.0, "speed": 1.8},
    "turbine":    {"temperature": 45.0, "vibration": 1.0, "rpm": 1490.0, "efficiency": 94.0},
}

_stress: Dict[str, float] = {
    "EQ-001": 0.75,   
    "EQ-002": 0.30,   
    "EQ-003": 0.05,  
    "EQ-004": 0.45,   
    "EQ-005": 0.02,   
}

FAILURE_DELTA = {
    "compressor": {"temperature": +30, "vibration": +7,  "pressure": +40, "current": +15},
    "pump":       {"temperature": +25, "vibration": +6,  "flow_rate": -60, "pressure": +30},
    "motor":      {"temperature": +28, "vibration": +6.5,"current":  +20, "rpm": -300},
    "conveyor":   {"temperature": +22, "vibration": +5,  "belt_tension":-250,"speed":-0.6},
    "turbine":    {"temperature": +35, "vibration": +7.5,"rpm": -200, "efficiency": -12},
}

_latest: Dict[str, dict] = {}

_sim_t = 0


def simulate_sensors(eq_id: str, eq_type: str) -> dict:
    """Generate realistic sensor values based on injected stress level."""
    global _sim_t
    nominal  = NOMINAL_SENSORS[eq_type]
    deltas   = FAILURE_DELTA[eq_type]
    stress   = _stress.get(eq_id, 0.1)
        osc = math.sin(_sim_t * 0.05) * 0.03
    sensors = {}
    for sensor, base in nominal.items():
        delta   = deltas.get(sensor, 0)
        drift   = delta * (stress ** 1.5)
        noise   = random.gauss(0, abs(base) * 0.01)
        sensors[sensor] = round(base + drift + noise + base * osc, 3)
    return sensors
class SensorReading(BaseModel):
    sensors: Dict[str, float]
    timestamp: Optional[str] = None
class PredictionResponse(BaseModel):
    equipment_id: str
    equipment_type: str
    timestamp: str
    sensors: Dict[str, float]
    anomaly_score: float
    failure_prob: float
    failure_prob_pct: float
    predicted_rul: Optional[float]
    health_score: float
    alert_level: str
    recommendations: List[str]
@app.get("/health")
def health():
    return {"status": "ok", "timestamp": datetime.utcnow().isoformat()}


@app.get("/equipment")
def list_equipment():
    return {"equipment": EQUIPMENT}


@app.post("/predict/{equipment_id}", response_model=PredictionResponse)
def run_prediction(equipment_id: str, reading: SensorReading):
    eq = next((e for e in EQUIPMENT if e["id"] == equipment_id), None)
    if not eq:
        raise HTTPException(status_code=404, detail=f"Equipment {equipment_id} not found")

    result = predict(equipment_id, eq["type"], reading.sensors)
    if "error" in result:
        raise HTTPException(status_code=500, detail=result["error"])

    out = {
        **result,
        "timestamp": reading.timestamp or datetime.utcnow().isoformat(),
        "sensors": reading.sensors,
    }
    _latest[equipment_id] = out
    return out


@app.get("/status/{equipment_id}")
def get_status(equipment_id: str):
    if equipment_id not in _latest:
        raise HTTPException(status_code=404, detail="No data yet for this equipment")
    return _latest[equipment_id]


@app.get("/alerts")
def get_alerts():
    alerts = []
    for eq_id, data in _latest.items():
        if data["alert_level"] in ("critical", "warning"):
            alerts.append({
                "equipment_id": eq_id,
                "alert_level": data["alert_level"],
                "failure_prob_pct": data["failure_prob_pct"],
                "health_score": data["health_score"],
                "predicted_rul": data["predicted_rul"],
                "recommendations": data["recommendations"],
                "timestamp": data["timestamp"],
            })
    alerts.sort(key=lambda a: a["failure_prob_pct"], reverse=True)
    return {"alerts": alerts, "count": len(alerts)}


@app.get("/fleet")
def fleet_summary():
    summary = []
    for eq in EQUIPMENT:
        latest = _latest.get(eq["id"], {})
        summary.append({
            "id": eq["id"],
            "name": eq["name"],
            "type": eq["type"],
            "health_score": latest.get("health_score", 100),
            "alert_level": latest.get("alert_level", "unknown"),
            "failure_prob_pct": latest.get("failure_prob_pct", 0),
            "predicted_rul": latest.get("predicted_rul"),
            "last_seen": latest.get("timestamp"),
        })
    return {"fleet": summary}

@app.websocket("/ws/live")
async def websocket_live(websocket: WebSocket):
    """
    Streams live simulated sensor readings + model predictions to connected clients.
    Sends a JSON object every 2 seconds covering all equipment.
    """
    await websocket.accept()
    global _sim_t
    try:
        while True:
            _sim_t += 1
            payload = {"timestamp": datetime.utcnow().isoformat(), "equipment": []}

            for eq in EQUIPMENT:
                sensors = simulate_sensors(eq["id"], eq["type"])
                result  = predict(eq["id"], eq["type"], sensors)
                _latest[eq["id"]] = {**result, "sensors": sensors,
                                      "timestamp": datetime.utcnow().isoformat()}
                payload["equipment"].append({
                    "id": eq["id"],
                    "name": eq["name"],
                    "type": eq["type"],
                    "sensors": sensors,
                    **result,
                })

            await websocket.send_text(json.dumps(payload))
            await asyncio.sleep(2)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.close()
        except Exception:
            pass
