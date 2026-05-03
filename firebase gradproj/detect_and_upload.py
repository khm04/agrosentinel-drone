import os
import uuid
from datetime import datetime, UTC

import cv2

from firebase_config import initialize_firebase

# Initialize once
db, bucket = initialize_firebase()


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_crop(image, bbox, output_dir="detections_local"):
    """
    Save cropped region from image using bbox = (x1, y1, x2, y2).
    """
    ensure_dir(output_dir)

    if image is None:
        raise ValueError("Input image is None.")

    height, width = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Clamp bbox to image boundaries
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(0, min(x2, width))
    y2 = max(0, min(y2, height))

    if x2 <= x1 or y2 <= y1:
        raise ValueError("Invalid bounding box: crop area is empty.")

    crop = image[y1:y2, x1:x2]

    filename = f"{uuid.uuid4().hex}.jpg"
    local_path = os.path.join(output_dir, filename)

    success = cv2.imwrite(local_path, crop)
    if not success:
        raise IOError("Failed to save cropped image.")

    return local_path, filename


def save_full_image(image, output_dir="detections_local"):
    """
    Save full image locally.
    """
    ensure_dir(output_dir)

    if image is None:
        raise ValueError("Input image is None.")

    filename = f"{uuid.uuid4().hex}.jpg"
    local_path = os.path.join(output_dir, filename)

    success = cv2.imwrite(local_path, image)
    if not success:
        raise IOError("Failed to save image.")

    return local_path, filename


def upload_image(local_path, anomaly_type):
    """
    Upload local file to Firebase Storage.
    Returns storage path and public URL if available.
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(f"Local file does not exist: {local_path}")

    today = datetime.now(UTC).strftime("%Y-%m-%d")
    filename = os.path.basename(local_path)

    storage_path = f"detections/{anomaly_type}/{today}/{filename}"

    blob = bucket.blob(storage_path)
    blob.upload_from_filename(local_path)

    return storage_path, storage_path


def log_event(anomaly_type, confidence, gps_lat, gps_lng, image_url, storage_path, extra=None):
    """
    Write detection event to Firestore.
    """
    data = {
        "anomaly_type": anomaly_type,
        "confidence": float(confidence),
        "gps_lat": float(gps_lat),
        "gps_lng": float(gps_lng),
        "timestamp": datetime.now(UTC).isoformat(),
        "image_url": image_url,
        "storage_path": storage_path,
        "status": "new",
    }

    if extra:
        data.update(extra)

    doc_ref = db.collection("events").add(data)
    return doc_ref


def handle_disease_classification(frame, disease_name, confidence, gps_lat, gps_lng, bbox=None):
    """
    Handle plant disease result.
    If bbox is provided, upload cropped detection.
    Otherwise upload full frame.
    """
    try:
        if bbox is not None:
            local_path, filename = save_crop(frame, bbox)
            saved_mode = "crop"
        else:
            local_path, filename = save_full_image(frame)
            saved_mode = "full_image"

        storage_path, image_url = upload_image(local_path, "disease")

        extra = {
            "source_file": filename,
            "disease_name": disease_name,
            "saved_mode": saved_mode,
        }

        if bbox is not None:
            extra["bbox"] = [int(v) for v in bbox]

        log_event(
            anomaly_type="disease",
            confidence=confidence,
            gps_lat=gps_lat,
            gps_lng=gps_lng,
            image_url=image_url,
            storage_path=storage_path,
            extra=extra,
        )

        print(f"✅ Disease uploaded successfully: {disease_name}")
        print(f"Storage path: {storage_path}")
        print(f"Image URL: {image_url}")

    except Exception as e:
        print(f"Error while handling disease classification: {e}")