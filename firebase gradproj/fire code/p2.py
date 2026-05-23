import torch
import os
import json
from ultralytics import YOLO
from datetime import datetime


# ── Dataset ────────────────────────────────────────────────────
DATASET         = "datasets/fire-8/data.yaml"

# ── Model ──────────────────────────────────────────────────────
MODEL           = "yolov8s.pt"

# ── Staged Training ────────────────────────────────────────────
FREEZE_EPOCHS   = 15
FREEZE_LAYERS   = 10
FINETUNE_EPOCHS = 150

# ── Core ───────────────────────────────────────────────────────
IMG_SIZE        = 640
WORKERS         = 4

# ── Optimizer ──────────────────────────────────────────────────
LR_PHASE1       = 0.001
LR_PHASE2       = 0.01
LRF             = 0.01
MOMENTUM        = 0.937
WEIGHT_DECAY    = 0.0005
WARMUP_EPOCHS   = 3
WARMUP_MOMENTUM = 0.8
WARMUP_BIAS_LR  = 0.1
PATIENCE        = 30

# ── Augmentation ───────────────────────────────────────────────
FLIPLR          = 0.5
FLIPUD          = 0.0
DEGREES         = 10.0
TRANSLATE       = 0.1
SCALE           = 0.5
SHEAR           = 2.0
PERSPECTIVE     = 0.0001
HSV_H           = 0.015
HSV_S           = 0.7
HSV_V           = 0.4
MOSAIC          = 1.0
CLOSE_MOSAIC    = 20
MIXUP           = 0.15
COPY_PASTE      = 0.1

# ── Loss ───────────────────────────────────────────────────────
BOX             = 7.5
CLS             = 0.5
DFL             = 1.5

# ── Speed & Memory ─────────────────────────────────────────────
CACHE           = "disk"
AMP             = True

# ── Save & Output ──────────────────────────────────────────────
PROJECT         = "training_runs"
SAVE_PERIOD     = 10
CONF            = 0.25

# ── Resume control ─────────────────────────────────────────────
# Set FORCE_RESTART = True to ignore any saved state and train from scratch.
FORCE_RESTART   = False

STATE_FILE      = os.path.join(PROJECT, "training_state.json")


# ╔══════════════════════════════════════════════════════════════╗
# ║                        HELPERS                               ║
# ╚══════════════════════════════════════════════════════════════╝

def detect_device():
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  GPU  : {name}")
        print(f"   VRAM : {vram:.1f} GB")
        if vram < 4:
            print("     Low VRAM — consider MODEL='yolov8n.pt' and BATCH=8")
        return "0", int(vram)
    else:
        print("  No GPU — running on CPU (slow)")
        return "cpu", 0

def auto_batch(vram_gb):
    if   vram_gb >= 16: return 64
    elif vram_gb >= 10: return 32
    elif vram_gb >=  6: return 16
    elif vram_gb >=  4: return 8
    else:               return 4

def shared_args(device, batch):
    return dict(
        data            = DATASET,
        imgsz           = IMG_SIZE,
        batch           = batch,
        device          = device,
        workers         = WORKERS,
        momentum        = MOMENTUM,
        weight_decay    = WEIGHT_DECAY,
        warmup_epochs   = WARMUP_EPOCHS,
        warmup_momentum = WARMUP_MOMENTUM,
        warmup_bias_lr  = WARMUP_BIAS_LR,
        patience        = PATIENCE,
        fliplr          = FLIPLR,
        flipud          = FLIPUD,
        degrees         = DEGREES,
        translate       = TRANSLATE,
        scale           = SCALE,
        shear           = SHEAR,
        perspective     = PERSPECTIVE,
        hsv_h           = HSV_H,
        hsv_s           = HSV_S,
        hsv_v           = HSV_V,
        mosaic          = MOSAIC,
        close_mosaic    = CLOSE_MOSAIC,
        mixup           = MIXUP,
        copy_paste      = COPY_PASTE,
        box             = BOX,
        cls             = CLS,
        dfl             = DFL,
        cache           = CACHE,
        amp             = AMP,
        plots           = True,
        verbose         = True,
        save_period     = SAVE_PERIOD,
        project         = PROJECT,
    )

def save_state(state: dict):
    os.makedirs(PROJECT, exist_ok=True)
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=2)

def load_state() -> dict | None:
    if os.path.exists(STATE_FILE):
        with open(STATE_FILE) as f:
            return json.load(f)
    return None

def validate(model, output_dir):
    print("\n Validating best model...")
    best = os.path.join(output_dir, "weights", "best.pt")
    if os.path.exists(best):
        model = YOLO(best)

    metrics = model.val(conf=CONF)
    map50   = metrics.box.map50
    map5095 = metrics.box.map
    prec    = metrics.box.mp
    rec     = metrics.box.mr
    f1      = 2 * prec * rec / (prec + rec + 1e-9)

    g = lambda v, t: "✅" if v >= t else ("⚠️ " if v >= t * 0.85 else "❌")

    print("\n" + "=" * 48)
    print("  FINAL RESULTS")
    print("=" * 48)
    print(f"  {g(map50,  0.85)}  mAP50     : {map50:.3f}   (target ≥ 0.85)")
    print(f"  {g(map5095,0.65)}  mAP50-95  : {map5095:.3f}   (target ≥ 0.65)")
    print(f"  {g(prec,   0.80)}  Precision : {prec:.3f}   (target ≥ 0.80)")
    print(f"  {g(rec,    0.80)}  Recall    : {rec:.3f}   (target ≥ 0.80)")
    print(f"       F1-Score  : {f1:.3f}")
    print("=" * 48)

    if map50 >= 0.85:
        print("\n Excellent model! Ready for deployment.")
    elif map50 >= 0.70:
        print("\n Good model. Consider more epochs or larger dataset.")
    else:
        print("\n  Low accuracy. Try more epochs or a larger model.")

    return metrics


# ╔══════════════════════════════════════════════════════════════╗
# ║                     CHECKPOINT DETECTION                     ║
# ╚══════════════════════════════════════════════════════════════╝

def detect_resume_state():
    """
    Returns a dict describing what to do on this run:
      mode        : 'fresh' | 'resume_phase1' | 'skip_to_phase2' | 'resume_phase2' | 'done'
      run_name    : the run name to use (existing or new)
      phase1_dir  : path to phase1 output dir
      phase2_dir  : path to phase2 output dir
      p1_last     : last.pt path for phase1 (if resuming phase1)
      p2_last     : last.pt path for phase2 (if resuming phase2)
      phase1_best : best.pt from phase1 (if skipping to phase2)
    """
    if FORCE_RESTART:
        print("  FORCE_RESTART=True — ignoring any saved checkpoints.")
        return _fresh_state()

    state = load_state()
    if state is None:
        return _fresh_state()

    run_name   = state["run_name"]
    phase1_dir = state["phase1_dir"]
    phase2_dir = state["phase2_dir"]

    # Phase 2 already finished
    if state.get("phase2_done"):
        return dict(mode="done", run_name=run_name,
                    phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                    p1_last=None, p2_last=None, phase1_best=None)

    # Phase 1 finished, phase 2 not done
    if state.get("phase1_done"):
        p2_last = os.path.join(phase2_dir, "weights", "last.pt")
        if os.path.exists(p2_last):
            return dict(mode="resume_phase2", run_name=run_name,
                        phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                        p1_last=None, p2_last=p2_last, phase1_best=None)
        else:
            phase1_best = os.path.join(phase1_dir, "weights", "best.pt")
            return dict(mode="skip_to_phase2", run_name=run_name,
                        phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                        p1_last=None, p2_last=None, phase1_best=phase1_best)

    # Phase 1 in progress
    p1_last = os.path.join(phase1_dir, "weights", "last.pt")
    if os.path.exists(p1_last):
        return dict(mode="resume_phase1", run_name=run_name,
                    phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                    p1_last=p1_last, p2_last=None, phase1_best=None)

    # State file exists but no weights found — start fresh
    print("  State file found but no weights on disk — starting fresh.")
    return _fresh_state()

def _fresh_state():
    run_name   = f"fire_{datetime.now().strftime('%Y%m%d_%H%M')}"
    phase1_dir = os.path.join(PROJECT, run_name + "_phase1")
    phase2_dir = os.path.join(PROJECT, run_name + "_phase2")
    return dict(mode="fresh", run_name=run_name,
                phase1_dir=phase1_dir, phase2_dir=phase2_dir,
                p1_last=None, p2_last=None, phase1_best=None)


# ╔══════════════════════════════════════════════════════════════╗
# ║                     STAGED TRAINING                          ║
# ╚══════════════════════════════════════════════════════════════╝

print("=" * 58)
print("   YOLOv8 Fire & Smoke — Best & Strongest Training")
print("=" * 58)

device, vram = detect_device()
batch        = auto_batch(vram)
os.makedirs(PROJECT, exist_ok=True)

rs = detect_resume_state()
mode       = rs["mode"]
run_name   = rs["run_name"]
phase1_dir = rs["phase1_dir"]
phase2_dir = rs["phase2_dir"]

print(f"\n  Model      : {MODEL}")
print(f"   Strategy   : STAGED (Phase1: frozen → Phase2: full)")
print(f"   Phase 1    : {FREEZE_EPOCHS} epochs  (AdamW, frozen backbone)")
print(f"   Phase 2    : {FINETUNE_EPOCHS} epochs  (SGD, all layers)")
print(f"   Img size   : {IMG_SIZE}px  |  Batch: {batch}  (auto)")
print(f"   Run name   : {run_name}")
print(f"   Resume mode: {mode}")

# ── Already finished ───────────────────────────────────────────
if mode == "done":
    print("\n  Training already complete for this run.")
    print(f"   Results in: {phase2_dir}/")
    print("   Set FORCE_RESTART=True to train from scratch.")
    validate(YOLO(os.path.join(phase2_dir, "weights", "best.pt")), phase2_dir)
    raise SystemExit(0)

# ── PHASE 1 ────────────────────────────────────────────────────
if mode in ("fresh", "resume_phase1"):
    print("\n" + "─" * 58)
    if mode == "resume_phase1":
        print(f"  RESUMING PHASE 1 — from {rs['p1_last']}")
    else:
        print(f"  PHASE 1 / 2 — Frozen backbone ({FREEZE_EPOCHS} epochs)")
        print(f"  Training detection head only with AdamW")
    print("─" * 58 + "\n")

    phase1_name = run_name + "_phase1"

    if mode == "resume_phase1":
        model = YOLO(rs["p1_last"])
        model.train(resume=True)
    else:
        model = YOLO(MODEL)
        model.train(
            **shared_args(device, batch),
            epochs    = FREEZE_EPOCHS,
            freeze    = FREEZE_LAYERS,
            optimizer = "AdamW",
            lr0       = LR_PHASE1,
            lrf       = LRF,
            name      = phase1_name,
        )

    save_state(dict(
        run_name    = run_name,
        phase1_dir  = phase1_dir,
        phase2_dir  = phase2_dir,
        phase1_done = True,
        phase2_done = False,
    ))
    phase1_best = os.path.join(phase1_dir, "weights", "best.pt")
    print(f"\n  Phase 1 done → {phase1_best}")
else:
    model = None  # phase1 already done, model2 will be loaded below

# ── PHASE 2 ────────────────────────────────────────────────────
print("\n" + "─" * 58)
if mode == "resume_phase2":
    print(f"  RESUMING PHASE 2 — from {rs['p2_last']}")
elif mode == "skip_to_phase2":
    print(f"  SKIPPING TO PHASE 2 — Phase 1 already complete")
    print(f"  Starting Phase 2 from {rs['phase1_best']}")
else:
    print(f"  PHASE 2 / 2 — Full fine-tune ({FINETUNE_EPOCHS} epochs)")
    print(f"  All layers unlocked, SGD with cosine LR decay")
print("─" * 58 + "\n")

phase2_name = run_name + "_phase2"

if mode == "resume_phase2":
    model2 = YOLO(rs["p2_last"])
    model2.train(resume=True)
elif mode == "skip_to_phase2":
    model2 = YOLO(rs["phase1_best"])
    model2.train(
        **shared_args(device, batch),
        epochs    = FINETUNE_EPOCHS,
        freeze    = 0,
        optimizer = "SGD",
        lr0       = LR_PHASE2,
        lrf       = LRF,
        name      = phase2_name,
    )
else:
    # fresh run — phase1 just finished, model is available
    phase1_best = os.path.join(phase1_dir, "weights", "best.pt")
    model2 = YOLO(phase1_best)
    model2.train(
        **shared_args(device, batch),
        epochs    = FINETUNE_EPOCHS,
        freeze    = 0,
        optimizer = "SGD",
        lr0       = LR_PHASE2,
        lrf       = LRF,
        name      = phase2_name,
    )

save_state(dict(
    run_name    = run_name,
    phase1_dir  = phase1_dir,
    phase2_dir  = phase2_dir,
    phase1_done = True,
    phase2_done = True,
))

# ── VALIDATE ───────────────────────────────────────────────────
validate(model2, phase2_dir)

# ── SUMMARY ────────────────────────────────────────────────────
best_final = os.path.join(phase2_dir, "weights", "best.pt")
print(f"\n  Done! All files saved to: {phase2_dir}/")
print(f"   └── weights/best.pt       ← use this for inference")
print(f"   └── weights/last.pt")
print(f"   └── results.png           ← loss & mAP curves")
print(f"   └── confusion_matrix.png  ← per-class accuracy")
print(f"\n  Update your test script:")
print(f'   MODEL_PATH = r"{os.path.abspath(best_final)}"')
