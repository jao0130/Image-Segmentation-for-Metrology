"""
Animal eye detection using a trained YOLOv8-pose model.

Replaces the HoughCircles approach in src/eye_detect.py once a trained
model is available at EYE_MODEL_PATH (set in .env).

The trained model predicts two keypoints per animal detection:
  kp[0] = left_eye   (x, y, confidence)
  kp[1] = right_eye  (x, y, confidence)

Integration with main.py
------------------------
In main.py, replace:
    from eye_detect import detect_eyes
    eyes = detect_eyes(image, animal["mask"], animal["bbox"])
with:
    from keypoint_model.predict import load_pose_model, detect_eyes_pose
    pose_model = load_pose_model()          # once, outside the loop
    eyes = detect_eyes_pose(pose_model, image, animal["bbox"])

The returned EyePair is identical in interface to the HoughCircles version,
so no other changes are needed.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO

# Keypoint indices as defined in prepare_dataset.py / dataset.yaml
KP_LEFT_EYE  = 0
KP_RIGHT_EYE = 1

DEFAULT_MODEL_PATH = os.getenv("EYE_MODEL_PATH",
                                "runs/keypoint/animal_eyes_v1/weights/best.pt")


@dataclass
class EyePair:
    left:       Optional[tuple]   # (x, y) in full-image pixel coords, or None
    right:      Optional[tuple]   # (x, y) in full-image pixel coords, or None
    method:     str               # "pose" | "fallback"
    conf_left:  float = field(default=0.0)
    conf_right: float = field(default=0.0)


def load_pose_model(model_path: str = DEFAULT_MODEL_PATH) -> YOLO:
    """
    Load the trained YOLOv8-pose model.

    Raises FileNotFoundError with a clear message if the weights are missing
    so the user knows to train first.
    """
    if not Path(model_path).exists():
        raise FileNotFoundError(
            f"Pose model not found: {model_path}\n\n"
            "Train the model first:\n"
            "  1. python src/keypoint_model/prepare_dataset.py\n"
            "  2. python src/keypoint_model/train.py\n\n"
            "Then set EYE_MODEL_PATH in .env to the best.pt path."
        )
    return YOLO(model_path)


def detect_eyes_pose(
    model:    YOLO,
    image:    np.ndarray,
    bbox:     tuple,
    det_conf: float = 0.25,
    kp_conf:  float = 0.30,
) -> EyePair:
    """
    Detect animal eyes with the trained YOLOv8-pose model.

    Args:
        model:    Loaded pose model (from load_pose_model())
        image:    Full BGR image array (H, W, 3)
        bbox:     (x1, y1, x2, y2) bounding box of the animal in image coords
        det_conf: Minimum detection confidence for the animal bounding box
        kp_conf:  Minimum keypoint confidence to accept a keypoint as valid

    Returns:
        EyePair — left/right may be None if confidence is too low
    """
    x1, y1, x2, y2 = bbox

    # Reject degenerate boxes
    if x2 <= x1 or y2 <= y1:
        return EyePair(left=None, right=None, method="fallback")

    # Pad ROI slightly to give the model context around the animal
    h, w = image.shape[:2]
    pad = max(int((x2 - x1) * 0.05), int((y2 - y1) * 0.05), 5)
    rx1, ry1 = max(0, x1 - pad), max(0, y1 - pad)
    rx2, ry2 = min(w, x2 + pad), min(h, y2 + pad)

    roi = image[ry1:ry2, rx1:rx2]

    results = model(roi, conf=det_conf, verbose=False)[0]

    if results.keypoints is None or len(results.keypoints.data) == 0:
        return EyePair(left=None, right=None, method="fallback")

    # Pick the detection with highest bounding-box confidence
    best = int(results.boxes.conf.argmax())
    kps = results.keypoints.data[best]   # shape (2, 3): [[x, y, conf], ...]

    left_pt,  left_c  = _extract_kp(kps, KP_LEFT_EYE,  kp_conf, rx1, ry1)
    right_pt, right_c = _extract_kp(kps, KP_RIGHT_EYE, kp_conf, rx1, ry1)

    if left_pt is None and right_pt is None:
        return EyePair(left=None, right=None, method="fallback")

    # Ensure geometric left/right ordering
    if left_pt and right_pt and left_pt[0] > right_pt[0]:
        left_pt,  right_pt  = right_pt,  left_pt
        left_c,   right_c   = right_c,   left_c

    return EyePair(
        left=left_pt,
        right=right_pt,
        method="pose",
        conf_left=left_c,
        conf_right=right_c,
    )


def _extract_kp(
    kps:     "torch.Tensor",
    idx:     int,
    min_conf: float,
    offset_x: int,
    offset_y: int,
) -> tuple[Optional[tuple], float]:
    """
    Extract one keypoint from the (N, 3) tensor, converting ROI→image coords.

    Returns (point_or_None, confidence).
    """
    kx, ky, kc = float(kps[idx][0]), float(kps[idx][1]), float(kps[idx][2])
    if kc < min_conf:
        return None, 0.0
    return (int(offset_x + kx), int(offset_y + ky)), kc
