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
CONF_UPLOAD   = 0.80
CONFIRM_SECS  = 2.0
SAVE_COOLDOWN = 30.0

# ── Fire model (object detection, 640px) ──────────────────────────
FIRE_CONF_DISPLAY = 0.35   # show box at this confidence
FIRE_IMGSZ        = 640
FIRE_IGNORE       = {"default"}
FIRE_MAX_AREA     = 0.65   # reject boxes covering > 65% of frame (background walls)
FIRE_EDGE_MARGIN  = 2      # reject boxes that bleed from the very frame edge
FIRE_STICKY_FRAMES = 6     # keep showing last box for N frames after detection lost

# ── Plant model (classification, 256px) ───────────────────────────
PLANT_CONF_DISPLAY = 0.60
PLANT_IMGSZ        = 256
PLANT_GREEN_THRESH = 0.08

# ── GPS ────────────────────────────────────────────────────────────
GPS_LAT = 32.0123
GPS_LNG = 36.1234

# ── Shared: raw camera frame ──────────────────────────────────────
_raw_frame    = None
_raw_frame_id = 0
_raw_lock     = threading.Lock()

# ── Shared: annotation data (workers write, main draws) ───────────
_fire_boxes  = []      # list of [x1, y1, x2, y2, label, conf]
_fire_alock  = threading.Lock()

# [0] = (label, conf, veg_bbox) or None
# veg_bbox = (x1, y1, x2, y2) of largest green region, or None
_plant_state = [None]
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
# FIRE WORKER
# ─────────────────────────────────────────────────────────────────
def _fire_worker():
    print("[fire] Loading model...")
    model = YOLO(str(MODEL_FIRE))
    print(f"[fire] Ready. Classes: {list(model.names.values())}")

    last_seen_id    = -1
    first_detected  = None
    last_uploaded   = 0.0
    missed_frames   = 0      # frames since last valid detection
    last_best_box   = None   # last confirmed best box for sticky display

    while not _stop.is_set():
        with _raw_lock:
            fid   = _raw_frame_id
            frame = _raw_frame.copy() if _raw_frame is not None else None

        if frame is None or fid == last_seen_id:
            time.sleep(0.005)
            continue
        last_seen_id = fid

        fh, fw     = frame.shape[:2]
        frame_area = fw * fh

        results = model(frame, conf=FIRE_CONF_DISPLAY, imgsz=FIRE_IMGSZ, verbose=False)
        result  = results[0]

        valid = []
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label  = result.names[cls_id]
            if label.lower() in FIRE_IGNORE:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            if (x2 - x1) * (y2 - y1) / frame_area > FIRE_MAX_AREA:
                continue
            if x1 <= FIRE_EDGE_MARGIN or x2 >= (fw - FIRE_EDGE_MARGIN):
                continue
            valid.append([x1, y1, x2, y2, label, conf])

        if valid:
            # Show only the single highest-confidence box — cleaner display
            best_box     = max(valid, key=lambda b: b[5])
            last_best_box = best_box
            missed_frames = 0
            boxes_data    = [best_box]
        else:
            missed_frames += 1
            # Keep the last box visible for FIRE_STICKY_FRAMES to absorb movement
            if missed_frames <= FIRE_STICKY_FRAMES and last_best_box is not None:
                boxes_data = [last_best_box]
            else:
                boxes_data    = []
                last_best_box = None

        with _fire_alock:
            _fire_boxes.clear()
            _fire_boxes.extend(boxes_data)

        now = time.time()

        # Only reset timer once sticky buffer is exhausted
        if not valid and missed_frames > FIRE_STICKY_FRAMES:
            if first_detected is not None:
                print("[fire] Lost — timer reset")
            first_detected = None
            continue

        if not valid:
            # Still in sticky window — don't advance the timer
            continue

        best = max(valid, key=lambda b: b[5])
        x1, y1, x2, y2, best_label, best_conf = best
        bbox = [x1, y1, x2, y2]

        if first_detected is None:
            first_detected = now
            print(f"[fire] Detected {best_label} {best_conf:.0%} — confirming...")

        if best_conf < CONF_UPLOAD:
            elapsed = now - first_detected
            print(f"[fire] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s (conf {best_conf:.0%})")
            continue

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
# PLANT WORKER
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

        # Vegetation gate
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

        # Find bounding box of largest green region to show on display
        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        veg_bbox = None
        if contours:
            largest  = max(contours, key=cv2.contourArea)
            gx, gy, gw, gh = cv2.boundingRect(largest)
            veg_bbox = (gx, gy, gx + gw, gy + gh)

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

        with _plant_alock:
            _plant_state[0] = (label, top1_conf, veg_bbox)

        if first_detected is None:
            first_detected = now
            print(f"[plant] Detected {label} {top1_conf:.0%} — confirming...")

        if top1_conf < CONF_UPLOAD:
            elapsed = now - first_detected
            print(f"[plant] Confirming... {elapsed:.1f}s / {CONFIRM_SECS}s (conf {top1_conf:.0%})")
            continue

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
    # Fire: red bounding boxes
    with _fire_alock:
        boxes = list(_fire_boxes)

    for x1, y1, x2, y2, label, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 60, 255), 2)
        cv2.putText(frame, f"{label} {conf:.2f}",
                    (x1, max(y1 - 8, 12)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 60, 255), 2)

    # Plant: green box around detected vegetation + label banner
    with _plant_alock:
        plant = _plant_state[0]

    if plant is not None:
        label, conf, veg_bbox = plant

        # Green box around the plant region
        if veg_bbox is not None:
            px1, py1, px2, py2 = veg_bbox
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 200, 60), 2)

        # Label banner top-left
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
