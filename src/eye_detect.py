from dataclasses import dataclass
from typing import Optional
import cv2
import numpy as np


@dataclass
class EyePair:
    left: Optional[tuple]   # (x, y) in original image coords, or None
    right: Optional[tuple]  # (x, y) in original image coords, or None
    method: str             # "hough" or "fallback"


def detect_eyes(image: np.ndarray, mask: np.ndarray, bbox: tuple) -> EyePair:
    """
    Detect left/right eye within a segmented animal.

    Args:
        image: Full BGR image (H, W, 3)
        mask:  Binary mask same size as image; 1 = animal pixel
        bbox:  (x1, y1, x2, y2) bounding box in image coordinates

    Returns:
        EyePair with coordinates in original image space
    """
    x1, y1, x2, y2 = bbox
    h = max(y2 - y1, 1)
    w = max(x2 - x1, 1)

    # Guard against degenerate bbox
    if x2 <= x1 or y2 <= y1:
        return EyePair(left=None, right=None, method="fallback")

    # Crop ROI and mask
    roi = image[y1:y2, x1:x2].copy()
    mask_roi = mask[y1:y2, x1:x2]

    # White-out background
    roi[mask_roi == 0] = 255

    # Upper 60% = head region
    head_h = max(int(h * 0.6), 1)
    roi_head = roi[:head_h, :]
    mask_head = mask_roi[:head_h, :]

    # Grayscale + blur
    gray = cv2.cvtColor(roi_head, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    min_r = max(2, int(h * 0.02))
    max_r = max(min_r + 1, int(h * 0.08))

    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1,
        minDist=min_r * 2, param1=50, param2=15,
        minRadius=min_r, maxRadius=max_r,
    )

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        # Keep circles inside the animal mask
        valid = [
            c for c in circles
            if 0 <= c[1] < head_h and 0 <= c[0] < w and mask_head[c[1], c[0]] == 1
        ]
        # Pick the 2 darkest
        valid.sort(key=lambda c: int(gray[min(c[1], head_h - 1), min(c[0], w - 1)]))
        if len(valid) >= 2:
            pts = sorted(valid[:2], key=lambda c: c[0])
            return EyePair(
                left=(int(x1 + pts[0][0]), int(y1 + pts[0][1])),
                right=(int(x1 + pts[1][0]), int(y1 + pts[1][1])),
                method="hough",
            )

    return _fallback_eyes(gray, mask_head, x1, y1, head_h, w)


def _fallback_eyes(gray: np.ndarray, mask_head: np.ndarray,
                   x1: int, y1: int, head_h: int, w: int) -> EyePair:
    """Find two darkest separated points in head region."""
    masked = gray.copy()
    masked[mask_head == 0] = 255

    sorted_idx = np.argsort(masked.flatten())
    candidates = []
    for idx in sorted_idx:
        py, px = divmod(int(idx), w)
        if not candidates or min(abs(px - c[0]) for c in candidates) > w * 0.1:
            candidates.append((px, py))
        if len(candidates) == 2:
            break

    if len(candidates) < 2:
        return EyePair(left=None, right=None, method="fallback")

    pts = sorted(candidates, key=lambda c: c[0])
    return EyePair(
        left=(x1 + pts[0][0], y1 + pts[0][1]),
        right=(x1 + pts[1][0], y1 + pts[1][1]),
        method="fallback",
    )
