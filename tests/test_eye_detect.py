import numpy as np
import cv2
import pytest
from src.eye_detect import detect_eyes, EyePair


def _make_image_with_eyes(left_pos=(25, 25), right_pos=(75, 25), size=100):
    """Synthetic BGR image: white bg, two dark circles as eyes."""
    img = np.full((size, size, 3), 200, dtype=np.uint8)
    mask = np.zeros((size, size), dtype=np.uint8)
    mask[5:size - 5, 5:size - 5] = 1
    cv2.circle(img, left_pos, 5, (20, 20, 20), -1)
    cv2.circle(img, right_pos, 5, (20, 20, 20), -1)
    bbox = (5, 5, size - 5, size - 5)
    return img, mask, bbox


def test_returns_eyepair():
    img, mask, bbox = _make_image_with_eyes()
    result = detect_eyes(img, mask, bbox)
    assert isinstance(result, EyePair)


def test_both_eyes_detected():
    img, mask, bbox = _make_image_with_eyes()
    result = detect_eyes(img, mask, bbox)
    assert result.left is not None
    assert result.right is not None


def test_left_eye_has_smaller_x():
    img, mask, bbox = _make_image_with_eyes(left_pos=(25, 25), right_pos=(75, 25))
    result = detect_eyes(img, mask, bbox)
    if result.left and result.right:
        assert result.left[0] < result.right[0]


def test_fallback_does_not_crash():
    """Uniform image (no circles detectable) should not raise."""
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    mask = np.ones((100, 100), dtype=np.uint8)
    bbox = (0, 0, 100, 100)
    result = detect_eyes(img, mask, bbox)
    assert result is not None
    assert result.method in ("hough", "fallback")


def test_coordinates_within_image_bounds():
    img, mask, bbox = _make_image_with_eyes()
    result = detect_eyes(img, mask, bbox)
    h, w = img.shape[:2]
    for pt in [result.left, result.right]:
        if pt is not None:
            assert 0 <= pt[0] < w
            assert 0 <= pt[1] < h
