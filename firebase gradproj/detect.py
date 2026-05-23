import cv2
import time
import threading
import numpy as np
from pathlib import Path
from ultralytics import YOLO

from detect_and_upload import handle_detection, handle_disease_classification

# ── Paths ──────────────────────────────────────────────────────────
SCRIPT_DIR  = Path(__file__).parent
MODEL_FIRE  = SCRIPT_DIR / "models" / "fire_best.pt"
MODEL_PLANT = SCRIPT_DIR / "models" / "plant_best.pt"

# ── Upload thresholds ──────────────────────────────────────────────
CONF_UPLOAD   = 0.80   # minimum confidence to trigger Firebase upload
CONFIRM_SECS  = 2.0    # seconds detection must hold before uploading
SAVE_COOLDOWN = 30.0   # seconds between uploads for the same model

# ── Fire model (object detection, 640px) ──────────────────────────
FIRE_CONF_DISPLAY = 0.25   # show box on screen at this confidence
FIRE_IMGSZ        = 640
FIRE_IGNORE       = {"default"}

# ── Plant model (classification, 256px) ───────────────────────────
PLANT_CONF_DISPLAY = 0.60   # show label on screen at this confidence
PLANT_IMGSZ        = 256
PLANT_GREEN_THRESH = 0.08   # minimum green pixel ratio to run plant model

# ── GPS (replace with real GPS module when ready) ─────────────────
GPS_LAT = 32.0123
GPS_LNG = 36.1234

# ── Shared state ──────────────────────────────────────────────────
_raw_frame    = None      # latest camera frame (BGR)
_raw_frame_id = 0         # monotonically increasing; workers skip unchanged frames
_raw_lock     = threading.Lock()

_annotated       = None   # frame with fire+plant drawings overlaid
_annotated_lock  = threading.Lock()

_stop = threading.Event()

def upload_to_firebase(model_name: str, label: str, confidence: float,
                       frame, bbox):
    try:
        if model_name == "fire":
            handle_detection(
                frame=frame,
                bbox=bbox,
                anomaly_type="fire",
                confidence=confidence,
                gps_lat=GPS_LAT,
                gps_lng=GPS_LNG,
                label=label,
            )
            print(f"[Firebase] fire uploaded: {label} {confidence:.0%}")
        elif model_name == "plant":
            handle_disease_classification(
                frame=frame,
                disease_name=label,
                confidence=confidence,
                gps_lat=GPS_LAT,
                gps_lng=GPS_LNG,
                bbox=None,   # plant model has no bbox — uploads full frame crop
            )
            print(f"[Firebase] plant uploaded: {label} {confidence:.0%}")
    except Exception as exc:
        print(f"[Firebase ERROR] {model_name}: {exc}")
