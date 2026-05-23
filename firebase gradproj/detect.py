import cv2
import time
import threading
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
FIRE_CONF_DISPLAY = 0.25
FIRE_IMGSZ        = 640
FIRE_IGNORE       = {"default"}

# ── Plant model (classification, 256px) ───────────────────────────
PLANT_CONF_DISPLAY = 0.60
PLANT_IMGSZ        = 256
PLANT_GREEN_THRESH = 0.08   # minimum green pixel ratio to run plant model

# ── GPS (replace with real GPS module when ready) ─────────────────
GPS_LAT = 32.0123
GPS_LNG = 36.1234

# ── Shared: raw camera frame ──────────────────────────────────────
# Main thread writes; workers read. Workers copy immediately under lock
# so inference runs outside the lock.
_raw_frame    = None
_raw_frame_id = 0
_raw_lock     = threading.Lock()

# ── Shared: annotation data (workers write, main draws) ───────────
# Workers store the RESULT of inference as plain data.
# Main thread draws this data onto the fresh live frame each cycle.
# This is what keeps the camera live and boxes tracking.
_fire_boxes  = []      # list of [x1, y1, x2, y2, label, conf]
_fire_alock  = threading.Lock()

_plant_state = [None]  # [0] = (label, conf) or None
_plant_alock = threading.Lock()

_stop = threading.Event()


# ─────────────────────────────────────────────────────────────────
# FIREBASE UPLOAD
# ─────────────────────────────────────────────────────────────────
def upload_to_firebase(model_name, label, confidence, frame, bbox):
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
                bbox=None,
            )
            print(f"[Firebase] plant uploaded: {label} {confidence:.0%}")
    except Exception as exc:
        print(f"[Firebase ERROR] {model_name}: {exc}")


# ─────────────────────────────────────────────────────────────────
# FIRE WORKER  (object detection — has bounding boxes)
# ─────────────────────────────────────────────────────────────────
def _fire_worker():
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

        results = model(frame, conf=FIRE_CONF_DISPLAY, imgsz=FIRE_IMGSZ, verbose=False)
        result  = results[0]

        boxes_data = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            if label.lower() in FIRE_IGNORE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            boxes_data.append([x1, y1, x2, y2, label, conf])

        # Store annotation data — main thread draws it on the live frame
        with _fire_alock:
            _fire_boxes.clear()
            _fire_boxes.extend(boxes_data)

        now = time.time()

        if not boxes_data:
            if first_detected is not None:
                print("[fire] Lost — timer reset")
            first_detected = None
            continue

        best = max(boxes_data, key=lambda b: b[5])
        x1, y1, x2, y2, best_label, best_conf = best
        bbox = [x1, y1, x2, y2]

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


# ─────────────────────────────────────────────────────────────────
# PLANT WORKER  (classification — NO bounding boxes)
# ─────────────────────────────────────────────────────────────────
def _plant_worker():
    print("[plant] Loading model...")
    model = YOLO(str(MODEL_PLANT))
    print(f"[plant] Ready. Classes: {list(model.names.values())}")

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

        # Vegetation gate — only classify if scene is green enough
        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = green_mask.sum() / 255 / (frame.shape[0] * frame.shape[1])

        if green_ratio < PLANT_GREEN_THRESH:
            with _plant_alock:
                _plant_state[0] = None
            if first_detected is not None:
                print("[plant] Not green enough — timer reset")
            first_detected = None
            continue

        results = model(frame, imgsz=PLANT_IMGSZ, verbose=False)
        result  = results[0]

        if result.probs is None:
            continue

        top1_idx  = int(result.probs.top1)
        top1_conf = float(result.probs.top1conf)
        label     = result.names[top1_idx]

        now = time.time()

        if top1_conf < PLANT_CONF_DISPLAY:
            with _plant_alock:
                _plant_state[0] = None
            if first_detected is not None:
                print("[plant] Low conf — timer reset")
            first_detected = None
            continue

        # Store annotation data — main thread draws it on the live frame
        with _plant_alock:
            _plant_state[0] = (label, top1_conf)

        if top1_conf < CONF_UPLOAD:
            first_detected = None
            continue

        if first_detected is None:
            first_detected = now
            print(f"[plant] Detected {label} {top1_conf:.0%} — confirming...")

        elapsed = now - first_detected
        if elapsed >= CONFIRM_SECS and (now - last_uploaded) >= SAVE_COOLDOWN:
            print(f"[plant] Confirmed {elapsed:.1f}s — uploading...")
            threading.Thread(
                target=upload_to_firebase,
                args=("plant", label, top1_conf, frame, None),
                daemon=True,
            ).start()
            last_uploaded  = now
            first_detected = None
        elif elapsed < CONFIRM_SECS:
            print(f"[plant] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s")


# ─────────────────────────────────────────────────────────────────
# DRAW  — called by main on a fresh frame every cycle
# ─────────────────────────────────────────────────────────────────
def _draw(frame):
    with _fire_alock:
        boxes = list(_fire_boxes)

    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

    with _plant_alock:
        plant = _plant_state[0]

    if plant is not None:
        label, conf = plant
        clean = label.replace("___", " ").replace("_", " ")
        text  = f"{clean}  {conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(frame, (8, 8), (8 + tw + 10, 8 + th + 12), (0, 0, 0), -1)
        cv2.putText(frame, text, (13, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 60), 2)


# ─────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────
def main():
    global _raw_frame, _raw_frame_id

    for name, path in [("Fire", MODEL_FIRE), ("Plant", MODEL_PLANT)]:
        if not path.exists():
            print(f"[ERROR] {name} model not found: {path}")
            return

    threading.Thread(target=_fire_worker,  daemon=True).start()
    threading.Thread(target=_plant_worker, daemon=True).start()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("[ERROR] Cannot open camera.")
        return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    print("\nAgroSentinel running — press Q to quit.")
    print(f"Upload threshold : {CONF_UPLOAD:.0%}")
    print(f"Confirm window   : {CONFIRM_SECS}s")
    print(f"Save cooldown    : {SAVE_COOLDOWN}s\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Camera read failed.")
            break

        with _raw_lock:
            _raw_frame     = frame.copy()
            _raw_frame_id += 1

        # Draw latest annotation data onto the fresh live frame
        _draw(frame)

        cv2.imshow("AgroSentinel", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
