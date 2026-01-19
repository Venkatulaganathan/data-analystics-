
"""
End-to-end YOLOv8 pipeline for X-ray dataset:
1. Train on your custom dataset defined in a YAML file.
2. Run inference on a folder of X-ray images.
3. Save annotated images.

Run from terminal (inside venv):
    python main_xray_yolov8.py
"""

import os
from pathlib import Path
from typing import Optional, Union, List

import cv2
from ultralytics import YOLO


# ================== USER CONFIG ==================

# Path to your dataset yaml (update this to your file)
DATA_YAML: str = "data.yaml"          # e.g. "data/xray.yaml"

# Base model to start from (good default for custom training)
BASE_MODEL: str = "yolov8n.pt"        # change to yolov8s.pt / m / l if you want

# Training hyperparameters
EPOCHS: int = 100
IMAGE_SIZE: int = 640
BATCH_SIZE: int = 8
DEVICE: Union[int, str] = 0          # 0 for first GPU, or "cpu" for CPU only

# Where to save training runs
PROJECT_NAME: str = "runs/xray_train"
RUN_NAME: str = "yolov8n_xray"

# Inference settings
# After training, use the best weights from this run
TRAINED_WEIGHTS_PATH: Optional[str] = None  # will be auto-filled if None

# Folder with X-ray images for inference (can be same as val or test images)
INFER_IMAGE_DIR: str = "inference_images"   # set to any folder with your X-rays

# Where to save annotated inference images
INFER_OUTPUT_DIR: str = "runs/xray_infer"

CONF_THRESHOLD: float = 0.25
IOU_THRESHOLD: float = 0.45

# =================================================


def ensure_file_exists(path: str, kind: str = "file"):
    p = Path(path)
    if kind == "file":
        if not p.is_file():
            raise FileNotFoundError(f"{kind.capitalize()} not found: {path}")
    elif kind == "dir":
        if not p.is_dir():
            raise FileNotFoundError(f"{kind.capitalize()} not found: {path}")
    else:
        raise ValueError("kind must be 'file' or 'dir'")


def train_model() -> str:
    """
    Train YOLOv8 model on custom X-ray dataset using DATA_YAML.
    Returns path to best weights.
    """
    print("[INFO] Verifying dataset YAML...")
    ensure_file_exists(DATA_YAML, kind="file")

    print("[INFO] Loading base model:", BASE_MODEL)
    model = YOLO(BASE_MODEL)

    print("[INFO] Starting training...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project=PROJECT_NAME,
        name=RUN_NAME,
        pretrained=True,      # fine-tune from pretrained weights
        patience=20,          # early stopping
    )

    # Ultralytics saves best weights under project/name/weights/best.pt
    run_dir = Path(PROJECT_NAME) / RUN_NAME
    best_weights = run_dir / "weights" / "best.pt"

    if not best_weights.is_file():
        # Fallback: if structure changed or run name changed
        # try to find best.pt in subdirectories
        candidates: List[Path] = list(run_dir.rglob("best.pt"))
        if not candidates:
            raise FileNotFoundError(
                f"Could not find best.pt in {run_dir}. "
                "Check training output directories."
            )
        best_weights = candidates[0]

    print(f"[INFO] Training finished. Best weights: {best_weights}")
    return str(best_weights)


def run_inference(weights_path: str):
    """
    Run inference on all images in INFER_IMAGE_DIR using trained weights.
    Saves annotated images to INFER_OUTPUT_DIR.
    """
    print("[INFO] Verifying inference image directory...")
    ensure_file_exists(INFER_IMAGE_DIR, kind="dir")
    os.makedirs(INFER_OUTPUT_DIR, exist_ok=True)

    print("[INFO] Loading trained model:", weights_path)
    model = YOLO(weights_path)

    # Collect image paths (common formats)
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    image_paths = [
        p for p in Path(INFER_IMAGE_DIR).rglob("*")
        if p.suffix.lower() in exts
    ]

    if not image_paths:
        raise FileNotFoundError(
            f"No images found in {INFER_IMAGE_DIR}. ")
