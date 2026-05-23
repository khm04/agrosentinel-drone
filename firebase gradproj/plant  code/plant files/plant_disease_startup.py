"""


BEFORE RUNNING — CHECKLIST:
        
  2. Create & activate a virtual environment:
       python -m venv agrosentinel_env
       # Windows:
       agrosentinel_env\\Scripts\\activate
      
  3. Select interpreter in VS Code:
       Ctrl+Shift+P > Python: Select Interpreter > agrosentinel_env

  4. Install dependencies (see bottom of this file for commands).

  5. Run:
       python agrosentinel_plant_disease.py
       OR press ▶️ in VS Code

EXPECTED FOLDER STRUCTURE AFTER FIRST RUN:
  agrosentinel/
    ├── agrosentinel_plant_disease.py   ← this file
    ├── agrosentinel_env/               ← virtual environment
    ├── plantvillage_raw/               ← auto-downloaded dataset
    │     └── PlantVillage/
    │           ├── Tomato___healthy/
    │           ├── Bell_pepper___healthy/
    │           └── ...
    ├── plant_disease_yolo/             ← built YOLO dataset
    ├── outputs/                        ← training logs + weights
    ├── plant_disease_best.pt           ← best model (easy access)
    ├── training_results.png            ← training curves
    ├── prediction_result.png           ← sample prediction
    └── validation_metrics.txt          ← accuracy report
═══════════════════════════════════════════════════════════════════
"""

# ─────────────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────────────
import os
import shutil
import random
import matplotlib
matplotlib.use("Agg")   # Saves plots to file — no popup needed in VS Code
                        # Change to "TkAgg" if you want live windows
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from ultralytics import YOLO

print("✅ All imports successful")
print("=" * 60)


# ─────────────────────────────────────────────────────────────────
# STEP 0 — LOCAL DATASET SETUP
# ─────────────────────────────────────────────────────────────────
# Download the PlantVillage dataset manually from:
# https://www.kaggle.com/datasets/emmarex/plantdisease
#
# Extract it and place it like this:
#
#   agrosentinel/
#       ├── agrosentinel_plant_disease.py
#       └── plantvillage_raw/
#             └── PlantVillage/
#                   ├── Tomato___healthy/
#                   ├── Tomato___Late_blight/
#                   ├── Potato___healthy/
#                   └── ...
#
# IMPORTANT:
# The class folders must be inside PlantVillage/.

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

RAW_DOWNLOAD_DIR = os.path.join(BASE_DIR, "plantvillage_raw")

if not os.path.exists(RAW_DOWNLOAD_DIR):
    raise FileNotFoundError(
        f"\n[ERROR] Dataset folder not found:\n"
        f"  {RAW_DOWNLOAD_DIR}\n\n"
        f"Please manually download and extract the PlantVillage dataset there.\n"
    )

print(f"[DATASET] ✅ Using local dataset folder: {RAW_DOWNLOAD_DIR}")

# ─────────────────────────────────────────────────────────────────
# STEP 2 — AUTO-DETECT DATASET ROOT
# ─────────────────────────────────────────────────────────────────
# The PlantVillage dataset nests its images inside a subfolder.
# We walk the directory tree to find the folder that actually
# contains the class subfolders (e.g. "Tomato___healthy").

def find_dataset_root(search_root):
    """
    Walk the tree and return the directory that contains
    the most class subfolders (identified by '___' in their names).
    """
    best_dir   = None
    best_count = 0

    for dirpath, dirnames, _ in os.walk(search_root):
        class_dirs = [d for d in dirnames if "___" in d or "healthy" in d.lower()]
        if len(class_dirs) > best_count:
            best_count = len(class_dirs)
            best_dir   = dirpath

    return best_dir

print(f"\n[INFO] Locating PlantVillage class folders inside: {RAW_DOWNLOAD_DIR}")
RAW_DATASET_PATH = find_dataset_root(RAW_DOWNLOAD_DIR)

if RAW_DATASET_PATH is None:
    raise FileNotFoundError(
        "\n[ERROR] Could not find class folders inside the downloaded dataset.\n"
        "Expected folders like 'Tomato___healthy', 'Bell_pepper___healthy', etc.\n"
        f"Please check: {RAW_DOWNLOAD_DIR}\n"
        "You can also manually unzip the PlantVillage dataset there.\n"
    )

print(f"[AUTO-DETECT] ✅ Dataset root found: '{RAW_DATASET_PATH}'")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — SHOW ALL AVAILABLE CLASSES
# ─────────────────────────────────────────────────────────────────

all_classes = sorted([
    d for d in os.listdir(RAW_DATASET_PATH)
    if os.path.isdir(os.path.join(RAW_DATASET_PATH, d))
])

print(f"\n[INFO] All available classes ({len(all_classes)} total):\n")
for cls in all_classes:
    n = len([
        f for f in os.listdir(os.path.join(RAW_DATASET_PATH, cls))
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ])
    print(f"  {cls:<55} {n:>5} images")


# ─────────────────────────────────────────────────────────────────
# STEP 4 — CONFIGURATION
# ─────────────────────────────────────────────────────────────────
# ┌─────────────────────────────────────────────────────────────┐
# │  EDIT THIS SECTION to change classes, epochs, image count   │
# └─────────────────────────────────────────────────────────────┘

YOLO_DATASET_PATH = os.path.join(BASE_DIR, "plant_disease_yolo")
OUTPUTS_DIR       = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# ── Selected classes for this starter run ────────────────────
# Covers: Tomato (all diseases), Bell Pepper, Cucumber,
#         Corn (Maize), and Potato — as requested.
# Add/remove class names from the list printed above.

SELECTED_CLASSES = [
    # ── TOMATO (all available classes) ──
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",

    # ── BELL PEPPER ──
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",

    # ── POTATO ──
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
]

# ── Training hyperparameters ──────────────────────────────────
IMAGES_PER_CLASS = 800      # Increase to 800–1000+ for final training
TRAIN_SPLIT      = 0.8      # 80% train, 20% validation
IMG_SIZE         = 256      # Input resolution (256 for better leaf detail)
EPOCHS           = 75       # Starter: 25. Increase to 50–100 for final run
BATCH_SIZE       = 16       # Lower to 8 if you get out-of-memory errors
MODEL_SIZE       = "s"      # "s" = small (2× more accurate than nano)
                            # Options: n / s / m / l / x (small→extra-large)
PATIENCE         = 15       # Early stopping: stop if no improvement for N epochs

# ── Validate that every selected class exists on disk ─────────
print("\n[INFO] Validating selected classes against dataset...")
missing = []
for cls in SELECTED_CLASSES:
    path = os.path.join(RAW_DATASET_PATH, cls)
    if not os.path.isdir(path):
        missing.append(cls)
    else:
        n = len([f for f in os.listdir(path)
                 if f.lower().endswith((".jpg", ".jpeg", ".png"))])
        print(f"  [OK]  {cls:<55} ({n} images)")

if missing:
    print("\n[WARN] The following classes were NOT found in the dataset:")
    for m in missing:
        print(f"  ✗ '{m}'")
    print("\n  These will be skipped. Check spelling against the list above.")
    SELECTED_CLASSES = [c for c in SELECTED_CLASSES if c not in missing]

print(f"\n[INFO] Proceeding with {len(SELECTED_CLASSES)} classes.\n")


# ─────────────────────────────────────────────────────────────────
# STEP 5 — BUILD YOLO CLASSIFICATION DATASET
# ─────────────────────────────────────────────────────────────────
# YOLOv8 classification expects:
#   dataset/
#     train/
#       class_a/   image1.jpg  image2.jpg  ...
#       class_b/   ...
#     val/
#       class_a/   ...
#       class_b/   ...

def build_yolo_dataset(raw_path, out_path, classes, n_per_class, train_split):
    """
    Copy a random subset of images from the raw dataset into
    the YOLO classification folder structure.
    Skips rebuild if the folder already has files.
    """
    if os.path.exists(out_path):
        total = sum(len(files) for _, _, files in os.walk(out_path))
        if total > 0:
            print(f"[DATASET] ✅ YOLO dataset already built ({total} images). Skipping.")
            return
        else:
            print("[DATASET] ⚠️  Folder exists but is empty — rebuilding...")
            shutil.rmtree(out_path)

    print("[DATASET] Building YOLO classification dataset...")

    # Create all split/class directories
    for split in ["train", "val"]:
        for cls in classes:
            os.makedirs(os.path.join(out_path, split, cls), exist_ok=True)

    summary = []
    for cls in classes:
        src  = os.path.join(raw_path, cls)
        imgs = [
            f for f in os.listdir(src)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        ]

        if not imgs:
            print(f"  [SKIP]  '{cls}' — no images found, skipping.")
            continue

        random.seed(42)
        random.shuffle(imgs)
        selected   = imgs[:n_per_class]
        cut        = int(len(selected) * train_split)
        train_imgs = selected[:cut]
        val_imgs   = selected[cut:]

        for img in train_imgs:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(out_path, "train", cls, img)
            )
        for img in val_imgs:
            shutil.copy(
                os.path.join(src, img),
                os.path.join(out_path, "val", cls, img)
            )

        summary.append((cls, len(train_imgs), len(val_imgs)))
        print(f"  ✓  {cls:<55} {len(train_imgs)} train | {len(val_imgs)} val")

    print(f"\n[DATASET] ✅ Dataset ready — {len(summary)} classes processed.\n")


build_yolo_dataset(
    RAW_DATASET_PATH,
    YOLO_DATASET_PATH,
    SELECTED_CLASSES,
    IMAGES_PER_CLASS,
    TRAIN_SPLIT,
)

# ── Verify no folders are empty ────────────────────────────────
print("[VERIFY] Checking dataset structure...")
all_ok = True
for split in ["train", "val"]:
    for cls in SELECTED_CLASSES:
        folder = os.path.join(YOLO_DATASET_PATH, split, cls)
        n = len(os.listdir(folder)) if os.path.isdir(folder) else 0
        tag = "  [OK]  " if n > 0 else "  [!!]  "
        if n == 0:
            all_ok = False
            print(f"{tag} {split}/{cls} — {n} images  ← EMPTY!")
        # Quiet on success to keep output clean; uncomment below to see all:
        # else:
        #     print(f"{tag} {split}/{cls} — {n} images")

if not all_ok:
    raise RuntimeError(
        "\n[ERROR] Some class folders are empty.\n"
        "Delete the dataset folder and re-run:\n"
        "  import shutil; shutil.rmtree('plant_disease_yolo')\n"
    )
print("[VERIFY] ✅ All class folders are populated.\n")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — LOAD YOLOv8n AND TRAIN
# ─────────────────────────────────────────────────────────────────
# yolov8n-cls.pt (~6 MB) downloads automatically on first run.
# It is cached locally after that — no repeated downloads.

print("[TRAIN] Loading model...")
PRETRAINED_PT = os.path.join(BASE_DIR, "plant_disease_best.pt")
if os.path.exists(PRETRAINED_PT):
    print(f"[TRAIN] ✅ Found existing checkpoint — fine-tuning from: {PRETRAINED_PT}")
    model = YOLO(PRETRAINED_PT)
else:
    print(f"[TRAIN] No checkpoint found — starting from YOLOv8{MODEL_SIZE}-cls.pt (ImageNet pretrained)")
    model = YOLO(f"yolov8{MODEL_SIZE}-cls.pt")

print(f"\n[TRAIN] Starting training:")
print(f"        Classes   : {len(SELECTED_CLASSES)}")
print(f"        Epochs    : {EPOCHS}  (early stop patience: {PATIENCE})")
print(f"        Img size  : {IMG_SIZE}px")
print(f"        Batch     : {BATCH_SIZE}")
print(f"        Model     : YOLOv8{MODEL_SIZE}-cls\n")

results = model.train(
    data     = YOLO_DATASET_PATH,
    epochs   = EPOCHS,
    imgsz    = IMG_SIZE,
    batch    = BATCH_SIZE,
    name     = "plant_disease_agrosentinel",
    project  = OUTPUTS_DIR,
    patience = PATIENCE,
    verbose  = True,
    plots    = True,          # generates confusion matrix + accuracy curves
    save     = True,
    exist_ok = True,          # overwrite previous run with same name
    augment  = True,
    degrees  = 15,            # random rotation ±15°
    fliplr   = 0.5,           # horizontal flip 50%
    flipud   = 0.1,           # vertical flip 10%
    hsv_h    = 0.015,         # hue jitter (handles different lighting)
    hsv_s    = 0.5,           # saturation jitter
    hsv_v    = 0.3,           # brightness jitter
    mixup    = 0.1,           # blend pairs of images
)

print("\n[TRAIN] ✅ Training complete!")


# ─────────────────────────────────────────────────────────────────
# STEP 7 — COPY BEST MODEL TO EASY LOCATION
# ─────────────────────────────────────────────────────────────────

best_model_src  = os.path.join(
    OUTPUTS_DIR, "plant_disease_agrosentinel", "weights", "best.pt"
)
best_model_dst  = os.path.join(BASE_DIR, "plant_disease_best.pt")

if os.path.exists(best_model_src):
    shutil.copy(best_model_src, best_model_dst)
    print(f"\n[MODEL] ✅ Best model saved → {best_model_dst}")
else:
    print(f"\n[MODEL] ⚠️  Could not find best.pt at: {best_model_src}")
    print("        Check outputs/ folder manually.")


# ─────────────────────────────────────────────────────────────────
# STEP 8 — COPY TRAINING CURVES IMAGE
# ─────────────────────────────────────────────────────────────────

results_src = os.path.join(
    OUTPUTS_DIR, "plant_disease_agrosentinel", "results.png"
)
results_dst = os.path.join(BASE_DIR, "training_results.png")

if os.path.exists(results_src):
    shutil.copy(results_src, results_dst)
    print(f"[PLOT]  ✅ Training curves → {results_dst}")
    print("        Open in VS Code Explorer to view (click the file).")
else:
    print("[PLOT]  ⚠️  results.png not found — check OUTPUTS_DIR.")


# ─────────────────────────────────────────────────────────────────
# STEP 9 — EVALUATE ON VALIDATION SET
# ─────────────────────────────────────────────────────────────────

print("\n[EVAL]  Running validation on held-out images...")
val_metrics = model.val()

top1 = val_metrics.top1 * 100
top5 = val_metrics.top5 * 100

print(f"\n  ┌──────────────────────────────────────┐")
print(f"  │  Top-1 Accuracy : {top1:6.2f}%            │")
print(f"  │  Top-5 Accuracy : {top5:6.2f}%            │")
print(f"  └──────────────────────────────────────┘")

# ── Save metrics report ────────────────────────────────────────
metrics_path = os.path.join(BASE_DIR, "validation_metrics.txt")
with open(metrics_path, "w") as f:
    f.write("AgroSentinel — Plant Disease YOLOv8 Validation Report\n")
    f.write("=" * 54 + "\n")
    f.write(f"Model         : YOLOv8{MODEL_SIZE}-cls\n")
    f.write(f"Classes ({len(SELECTED_CLASSES):>2})   :\n")
    for cls in SELECTED_CLASSES:
        f.write(f"               • {cls}\n")
    f.write(f"Images/class  : {IMAGES_PER_CLASS}\n")
    f.write(f"Epochs run    : {EPOCHS}\n")
    f.write(f"Image size    : {IMG_SIZE}px\n")
    f.write(f"Batch size    : {BATCH_SIZE}\n")
    f.write(f"Top-1 Accuracy: {top1:.2f}%\n")
    f.write(f"Top-5 Accuracy: {top5:.2f}%\n")

print(f"\n[EVAL]  ✅ Metrics saved → {metrics_path}")


# ─────────────────────────────────────────────────────────────────
# STEP 10 — PREDICT A RANDOM VALIDATION IMAGE + SAVE AS PNG
# ─────────────────────────────────────────────────────────────────
# All results are saved as .png files — no popup windows.
# Open prediction_result.png in VS Code Explorer to inspect.

def predict_and_save(img_path, model, save_path):
    """
    Run inference on one image and save a side-by-side
    figure: (leaf photo | confidence bar chart) → PNG.
    """
    results    = model.predict(img_path, verbose=False)
    r          = results[0]
    probs      = r.probs
    top_idx    = int(probs.top1)
    top_conf   = float(probs.top1conf)
    class_name = r.names[top_idx]
    all_probs  = probs.data.cpu().numpy()
    class_names = [r.names[i] for i in range(len(all_probs))]
    colors      = [
        "#2ECC71" if i == top_idx else "#3498DB"
        for i in range(len(all_probs))
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, max(5, len(all_probs) * 0.45)))
    fig.patch.set_facecolor("#1A1A2E")

    # Left panel — leaf image
    leaf_img = mpimg.imread(img_path)
    axes[0].imshow(leaf_img)
    axes[0].axis("off")
    title_color = "#2ECC71" if "healthy" in class_name.lower() else "#E74C3C"
    axes[0].set_title(
        f"Prediction: {class_name}\nConfidence: {top_conf*100:.1f}%",
        fontsize=11, fontweight="bold", color=title_color, pad=10
    )
    axes[0].set_facecolor("#1A1A2E")

    # Right panel — horizontal bar chart
    axes[1].set_facecolor("#16213E")
    bars = axes[1].barh(class_names, all_probs * 100, color=colors, height=0.6)
    axes[1].set_xlabel("Confidence (%)", color="white")
    axes[1].set_title("All Class Probabilities", color="white")
    axes[1].set_xlim(0, 110)
    axes[1].tick_params(colors="white", labelsize=8)
    axes[1].spines[:].set_color("#334155")
    for i, v in enumerate(all_probs * 100):
        if v > 0.5:
            axes[1].text(v + 1, i, f"{v:.1f}%", va="center",
                         color="white", fontsize=7)

    plt.suptitle("AgroSentinel — Plant Disease Detection",
                 color="white", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()

    print(f"\n  [PREDICTION] → {class_name}  ({top_conf*100:.1f}% confidence)")
    print(f"  [RESULT IMG] → {save_path}")
    print(f"               Click it in VS Code Explorer to view.")


# Pick a random validation image from one of the disease classes
test_cls = next(
    (c for c in SELECTED_CLASSES if "healthy" not in c.lower()),
    SELECTED_CLASSES[0]
)
test_folder = os.path.join(YOLO_DATASET_PATH, "val", test_cls)

if os.path.isdir(test_folder) and os.listdir(test_folder):
    test_img   = os.path.join(test_folder, random.choice(os.listdir(test_folder)))
    result_img = os.path.join(BASE_DIR, "prediction_result.png")
    print(f"\n[PREDICT] Running on sample image: {test_img}")
    predict_and_save(test_img, model, result_img)
else:
    print(f"\n[PREDICT] ⚠️  Could not find images in {test_folder}")


# ─────────────────────────────────────────────────────────────────
# STEP 11 — EXPORT TO ONNX (for Jetson Nano — uncomment when ready)
# ─────────────────────────────────────────────────────────────────
#
# After training is finalized and you're ready for Jetson deployment:
#
#   best = YOLO("plant_disease_best.pt")
#   best.export(format="onnx", imgsz=224)
#
# Then on the Jetson Nano terminal:
#   trtexec --onnx=plant_disease_best.onnx \
#           --saveEngine=plant_disease.engine \
#           --fp16
#
# Load the TensorRT engine in your live inference pipeline:
#   model_trt = YOLO("plant_disease.engine")   # ultralytics supports this


# ─────────────────────────────────────────────────────────────────
# DONE — Output file summary
# ─────────────────────────────────────────────────────────────────
print("\n")
print("╔══════════════════════════════════════════════════════════╗")
print("║         AgroSentinel Training Complete! 🌱              ║")
print("╠══════════════════════════════════════════════════════════╣")
print(f"║  Output files in: {BASE_DIR[:38]:<38} ║")
print("╠══════════════════════════════════════════════════════════╣")
print("║  plant_disease_best.pt       ← deploy this model        ║")
print("║  training_results.png        ← training/accuracy curves ║")
print("║  prediction_result.png       ← sample prediction viz    ║")
print("║  validation_metrics.txt      ← top-1 / top-5 accuracy   ║")
print("║  outputs/plant_disease_*/    ← full Ultralytics logs     ║")
print("╠══════════════════════════════════════════════════════════╣")
print("║  NEXT STEPS:                                            ║")
print("║  • Increase IMAGES_PER_CLASS to 800+ for better mAP    ║")
print("║  • Increase EPOCHS to 50–100 for final training         ║")
print("║  • Add PlantDoc dataset for aerial augmentation         ║")
print("║  • Uncomment Step 11 when ready for Jetson Nano         ║")
print("╚══════════════════════════════════════════════════════════╝\n")
print("  To use this model in your live inference code:")
print("    from ultralytics import YOLO")
print("    model = YOLO('plant_disease_best.pt')")
print("    results = model.predict('your_image.jpg')\n")
