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


def _fire_worker():
    global _annotated

    print("[fire] Loading model...")
    model = YOLO(str(MODEL_FIRE))
    print(f"[fire] Ready. Classes: {list(model.names.values())}")

    last_seen_id   = -1
    first_detected = None
    last_uploaded  = 0.0

    while not _stop.is_set():
        with _raw_lock:
            fid   = _raw_frame_id
            frame = _raw_frame.copy() if _raw_frame is not None else None

        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        results = model(frame, conf=FIRE_CONF_DISPLAY, imgsz=FIRE_IMGSZ,
                        verbose=False)
        result  = results[0]

        # Start from latest annotated frame so plant labels are preserved
        with _annotated_lock:
            base = _annotated.copy() if _annotated is not None else frame.copy()

        valid_boxes = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            if label.lower() in FIRE_IGNORE:
                continue
            valid_boxes.append(box)

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cv2.rectangle(base, (x1, y1), (x2, y2), (0, 60, 255), 2)
            cv2.putText(base, f"{label} {conf:.2f}",
                        (x1, max(y1 - 8, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

        with _annotated_lock:
            _annotated = base

        now = time.time()

        if not valid_boxes:
            if first_detected is not None:
                print("[fire] Lost — timer reset")
            first_detected = None
            continue

        best_box  = max(valid_boxes, key=lambda b: float(b.conf[0]))
        best_conf = float(best_box.conf[0])
        cls_id    = int(best_box.cls[0])
        best_label = result.names[cls_id]
        bbox       = best_box.xyxy[0].tolist()

        if best_conf < CONF_UPLOAD:
            first_detected = None
            continue

        if first_detected is None:
            first_detected = now
            print(f"[fire] Detected {best_label} {best_conf:.0%} — confirming...")

        elapsed = now - first_detected
        if elapsed >= CONFIRM_SECS and (now - last_uploaded) >= SAVE_COOLDOWN:
            print(f"[fire] Confirmed {elapsed:.1f}s — uploading...")
            threading.Thread(
                target=upload_to_firebase,
                args=("fire", best_label, best_conf, frame, bbox),
                daemon=True,
            ).start()
            last_uploaded  = now
            first_detected = None
        elif elapsed < CONFIRM_SECS:
            print(f"[fire] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s")
