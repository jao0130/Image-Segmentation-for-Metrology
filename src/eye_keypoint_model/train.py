"""
Train YOLOv8-pose for animal eye keypoint detection.

Trains a YOLOv8-pose model on the AP-10K dataset (converted by prepare_dataset.py)
to predict left_eye and right_eye keypoints on animals.

Hardware recommendation (matches project specs):
    RTX 4060 Ti 16GB  →  batch=16, imgsz=640, model=yolov8n-pose or yolov8s-pose

Usage:
    python src/keypoint_model/train.py

    # Custom settings:
    python src/keypoint_model/train.py --epochs 200 --model yolov8s-pose.pt --batch 8
"""

import argparse
import shutil
from pathlib import Path

from tqdm import tqdm
from ultralytics import YOLO

# ── Compact progress display ───────────────────────────────────────────────────

_pbar: tqdm | None = None
_total_epochs = 0


def _on_train_start(trainer):
    global _pbar, _total_epochs
    _total_epochs = trainer.epochs
    cols = max(shutil.get_terminal_size().columns - 2, 40)
    _pbar = tqdm(total=_total_epochs, ncols=cols,
                 bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]",
                 desc="Training", colour="cyan")


def _on_train_epoch_end(trainer):
    if _pbar is None:
        return
    m = trainer.metrics
    cols     = max(shutil.get_terminal_size().columns - 2, 40)
    _pbar.ncols = cols

    box_loss  = getattr(trainer.loss_items, 'tolist', lambda: [0]*4)()
    pose_loss = box_loss[3] if len(box_loss) > 3 else 0.0
    map50p    = m.get("metrics/mAP50(P)", 0.0)
    map50b    = m.get("metrics/mAP50(B)", 0.0)

    _pbar.set_postfix({
        "pose_loss": f"{pose_loss:.3f}",
        "mAP50(P)":  f"{map50p:.3f}",
        "mAP50(B)":  f"{map50b:.3f}",
    }, refresh=False)
    _pbar.update(1)


def _on_train_end(trainer):
    if _pbar is not None:
        _pbar.close()

# Project root = two levels up from this file (src/eye_keypoint_model/train.py)
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default paths (absolute, independent of working directory)
DEFAULT_DATASET = str(PROJECT_ROOT / "data/ap10k_eye_keypoints/dataset.yaml")
DEFAULT_MODEL   = "yolov8s-pose.pt"
DEFAULT_OUTPUT  = str(PROJECT_ROOT / "runs/keypoint")
DEFAULT_NAME    = "animal_eyes_"


def train(
    dataset: str   = DEFAULT_DATASET,
    base_model: str = DEFAULT_MODEL,
    epochs: int    = 150,
    imgsz: int     = 640,
    batch: int     = 16,
    device: str    = "0",        # "0" = first GPU; "cpu" = CPU
    project: str   = DEFAULT_OUTPUT,
    name: str      = DEFAULT_NAME,
    resume: bool   = False,
) -> str:
    """
    Fine-tune YOLOv8-pose on animal eye keypoints.

    Returns path to best weights file.
    """
    if not Path(dataset).exists():
        raise FileNotFoundError(
            f"Dataset not found: {dataset}\n"
            "Run `python src/keypoint_model/prepare_dataset.py` first."
        )

    model = YOLO(base_model)

    model.add_callback("on_train_start",     _on_train_start)
    model.add_callback("on_train_epoch_end", _on_train_epoch_end)
    model.add_callback("on_train_end",       _on_train_end)

    model.train(
        data=dataset,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,

        # ── Loss weights ──────────────────────────────────────────────────
        # pose  : keypoint regression loss weight (increase for better eye loc)
        # kobj  : keypoint objectness loss weight
        pose=6.0,
        kobj=1.5,

        # ── Augmentation ──────────────────────────────────────────────────
        # Horizontal flip is fine for eyes (left/right are symmetric)
        fliplr=0.5,
        # Do NOT flip vertically — eyes are always in upper region
        flipud=0.0,
        # Colour jitter — helps generalise across animal fur colours
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Mosaic and mixup improve small object detection
        mosaic=1.0,
        mixup=0.1,

        # ── Training control ──────────────────────────────────────────────
        # Stop early if no improvement for 30 epochs
        patience=30,
        # Save checkpoint every N epochs
        save_period=10,
        # Workers: 80% of 12 threads = ~9, capped at 8 for safety
        workers=8,

        verbose=False,
    )

    best = Path(project) / name / "weights" / "best.pt"
    print(f"\nTraining complete.")
    print(f"Best weights : {best}")
    print(f"Usage        : set EYE_MODEL_PATH={best} in .env")
    return str(best)


def parse_args():
    p = argparse.ArgumentParser(description="Train animal eye keypoint model")
    p.add_argument("--dataset",   default=DEFAULT_DATASET)
    p.add_argument("--model",     default=DEFAULT_MODEL,
                   help="Base weights (yolov8n-pose.pt / yolov8s-pose.pt / yolov8m-pose.pt)")
    p.add_argument("--epochs",    type=int, default=100)
    p.add_argument("--imgsz",     type=int, default=640)
    p.add_argument("--batch",     type=int, default=16,
                   help="Reduce to 8 if GPU OOM")
    p.add_argument("--device",    default="0",
                   help="GPU index or 'cpu'")
    p.add_argument("--project",   default=DEFAULT_OUTPUT)
    p.add_argument("--name",      default=DEFAULT_NAME)
    p.add_argument("--resume",    action="store_true",
                   help="Resume from last checkpoint")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        dataset=args.dataset,
        base_model=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        resume=args.resume,
    )
