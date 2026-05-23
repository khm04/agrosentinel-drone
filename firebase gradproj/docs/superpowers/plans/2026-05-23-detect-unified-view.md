# Detect Unified View Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `detect.py` with a clean unified-view implementation that shows fire bboxes and plant disease labels on a single camera frame, with simplified fire filtering to restore accuracy.

**Architecture:** Two background worker threads (fire, plant) each run their model on every new camera frame and write their annotations onto a shared `annotated_frame` protected by a lock. The main thread captures frames, feeds the shared buffer, reads the annotated result, and displays it in one window. When either worker detects ≥80% confidence for 2 consecutive seconds, it fires a background upload thread that calls `detect_and_upload.py`.

**Tech Stack:** Python 3.10+, OpenCV (`cv2`), Ultralytics YOLOv8, threading, `detect_and_upload.py` (unchanged)

---

## File Map

| Action | Path | Responsibility |
|--------|------|----------------|
| Replace | `detect.py` | Camera loop, two inference workers, unified display, upload trigger |
| Keep as-is | `detect_and_upload.py` | Firebase Storage upload + Firestore write |
| Keep as-is | `firebase_config.py` | Firebase Admin SDK init |

---

### Task 1: Scaffold the new `detect.py` with constants and shared state

**Files:**
- Replace: `detect.py`

- [ ] **Step 1: Delete the old file and write the new scaffold**

Replace the entire contents of `detect.py` with:

```python
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
```

- [ ] **Step 2: Verify Python parses correctly**

Run:
```
cd "c:\Users\user\Desktop\every gradfile\grad project fire base and flutter\firebase gradproj"
python -c "import detect"
```
Expected: no output, no errors. (Models won't load — that's fine at import time.)

- [ ] **Step 3: Commit**

```bash
git add detect.py
git commit -m "feat: scaffold new detect.py with constants and shared state"
```

---

### Task 2: Add the `upload_to_firebase` helper

**Files:**
- Modify: `detect.py`

- [ ] **Step 1: Append the upload helper after the shared state block**

Add to `detect.py`:

```python
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
```

- [ ] **Step 2: Verify parse**

```
python -c "import detect"
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add detect.py
git commit -m "feat: add upload_to_firebase helper"
```

---

### Task 3: Add the fire worker

**Files:**
- Modify: `detect.py`

- [ ] **Step 1: Append the fire worker function**

Add to `detect.py`:

```python
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
```

- [ ] **Step 2: Verify parse**

```
python -c "import detect"
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add detect.py
git commit -m "feat: add fire inference worker with bbox drawing"
```

---

### Task 4: Add the plant worker

**Files:**
- Modify: `detect.py`

- [ ] **Step 1: Append the plant worker function**

Add to `detect.py`:

```python
def _plant_worker():
    global _annotated

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

        # Vegetation gate — skip non-plant scenes
        hsv         = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        green_mask  = cv2.inRange(hsv, (35, 40, 40), (85, 255, 255))
        green_ratio = green_mask.sum() / 255 / (frame.shape[0] * frame.shape[1])

        if green_ratio < PLANT_GREEN_THRESH:
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
            if first_detected is not None:
                print("[plant] Low conf — timer reset")
            first_detected = None
            continue

        # Draw label banner on top of the current annotated frame
        with _annotated_lock:
            base = _annotated.copy() if _annotated is not None else frame.copy()

        clean = label.replace("___", " ").replace("_", " ")
        text  = f"{clean}  {top1_conf:.0%}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
        cv2.rectangle(base, (8, 8), (8 + tw + 10, 8 + th + 12), (0, 0, 0), -1)
        cv2.putText(base, text, (13, 8 + th + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 200, 60), 2)

        with _annotated_lock:
            _annotated = base

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
```

- [ ] **Step 2: Verify parse**

```
python -c "import detect"
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add detect.py
git commit -m "feat: add plant inference worker with label banner"
```

---

### Task 5: Add the `main()` loop

**Files:**
- Modify: `detect.py`

- [ ] **Step 1: Append `main()` and the entry point**

Add to `detect.py`:

```python
def main():
    global _raw_frame, _raw_frame_id, _annotated

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

        # Seed annotated frame each cycle so workers always have a fresh base
        with _annotated_lock:
            if _annotated is None:
                _annotated = frame.copy()

        with _annotated_lock:
            display = _annotated.copy()

        cv2.imshow("AgroSentinel", display)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    _stop.set()
    cap.release()
    cv2.destroyAllWindows()
    print("Stopped.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify the full file parses**

```
python -c "import detect"
```
Expected: no errors.

- [ ] **Step 3: Commit**

```bash
git add detect.py
git commit -m "feat: add main camera loop — unified single-window display"
```

---

### Task 6: End-to-end smoke test (no Firebase)

**Files:**
- Read: `detect.py`

- [ ] **Step 1: Run with camera, confirm display**

```
python detect.py
```

Expected startup output:
```
[fire] Loading model...
[fire] Ready. Classes: ['fire', 'smoke', ...]
[plant] Loading model...
[plant] Ready. Classes: ['Bell_pepper___healthy', ...]
AgroSentinel running — press Q to quit.
```

Expected window: single camera feed labeled `AgroSentinel`. Fire detections show red bounding boxes. Plant disease shows green label banner in top-left.

- [ ] **Step 2: Verify fire bbox draws correctly**

Point camera at a phone screen showing a fire image or an open flame. Confirm a red rectangle appears around the fire with a confidence label.

- [ ] **Step 3: Verify plant label draws correctly**

Point camera at a green plant. Confirm a green text banner appears in the top-left corner with the disease/healthy classification.

- [ ] **Step 4: Verify both annotations appear on the same frame**

Point camera at a plant scene. Optionally show fire on another part of the frame. Confirm both a green label AND a red bbox appear simultaneously in the single window.

- [ ] **Step 5: Commit (no code change needed — this is observation only)**

If no code changes were needed, skip commit. If you fixed a display bug, commit with:
```bash
git add detect.py
git commit -m "fix: correct unified-view annotation overlap"
```

---

### Task 7: End-to-end Firebase upload test

**Files:**
- Read: `detect.py`, `detect_and_upload.py`

- [ ] **Step 1: Trigger a fire upload**

Hold a fire image in front of the camera for 3+ seconds at high confidence. Watch the terminal for:
```
[fire] Detected fire 85% — confirming...
[fire] Confirming... 1.0s / 2.0s
[fire] Confirmed 2.1s — uploading...
[Firebase] fire uploaded: fire 85%
✅ fire uploaded successfully: fire
Storage path: detections/fire/2026-05-23/<uuid>.jpg
```

- [ ] **Step 2: Verify Firestore document**

Open Firebase Console → Firestore → `events` collection. Confirm a new document appeared with fields:
```
anomaly_type: "fire"
confidence: 0.85
gps_lat: 32.0123
gps_lng: 36.1234
status: "new"
storage_path: "detections/fire/2026-05-23/<uuid>.jpg"
timestamp: "2026-05-23T..."
```

- [ ] **Step 3: Verify Flutter app receives the event**

Open the Flutter app. Confirm the new detection appears in the list instantly (Firestore real-time listener).

- [ ] **Step 4: Final commit**

```bash
git add detect.py
git commit -m "feat: complete unified detect.py — fire bbox + plant label + Firebase upload"
```
