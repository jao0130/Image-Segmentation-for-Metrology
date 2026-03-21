from pathlib import Path
import cv2
import numpy as np

PALETTE = [
    (0, 220, 0), (0, 80, 255), (255, 140, 0), (200, 0, 200),
    (0, 210, 210), (100, 0, 180), (0, 160, 255), (255, 100, 0),
]

COLOR_EYE       = (0,   0,   255)
COLOR_INTER_EYE = (0,   220, 220)
COLOR_CROSS     = (0,   140, 255)
COLOR_TEXT_BG   = (20,  20,  20)
COLOR_PANEL_BG  = (30,  30,  30)

FONT = cv2.FONT_HERSHEY_DUPLEX

# Reference base dimension: scale factors are relative to 640px
_BASE = 640.0


def _scale(img, base_val):
    """Scale a pixel/font value proportionally to the shorter image edge."""
    h, w = img.shape[:2]
    return max(base_val * min(h, w) / _BASE, base_val * 0.5)


def _fs(img, base_font):
    """Return font scale relative to image size."""
    return round(_scale(img, base_font), 2)


def _th(img):
    """Line/text thickness relative to image size."""
    return max(1, int(_scale(img, 1.5)))


def _put_text_with_bg(img, text, pos, font_scale, color, thickness=None, padding=None):
    """Draw text with a semi-transparent dark background box for readability."""
    if thickness is None:
        thickness = _th(img)
    if padding is None:
        padding = max(3, int(_scale(img, 4)))
    (tw, th), baseline = cv2.getTextSize(text, FONT, font_scale, thickness)
    x, y = pos
    rx1 = max(0, x - padding)
    ry1 = max(0, y - th - padding)
    rx2 = min(img.shape[1] - 1, x + tw + padding)
    ry2 = min(img.shape[0] - 1, y + baseline + padding)
    overlay = img.copy()
    cv2.rectangle(overlay, (rx1, ry1), (rx2, ry2), COLOR_TEXT_BG, -1)
    cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
    cv2.putText(img, text, (x, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def _draw_info_panel(img, animals):
    """Draw a scaled info panel in the top-left corner."""
    h, w = img.shape[:2]

    lines = ["  MEASUREMENTS"]
    for i, a in enumerate(animals):
        d      = a.get("inter_eye_dist")
        eyes   = a.get("eyes")
        method = eyes.method if eyes else "N/A"
        dist_str = f"{d:.1f}px" if d is not None else "N/A"
        lines.append(f"  [{i}] {a['class_name']:10s}  inter-eye: {dist_str:>9s}  ({method})")

    for i in range(len(animals)):
        for j in range(i + 1, len(animals)):
            d = animals[i].get(f"cross_{j}_dist")
            if d is not None:
                lines.append(f"  [{i}]<->[{j}] right-eye dist: {d:.1f}px")

    font_scale = _fs(img, 0.48)
    thickness  = _th(img)
    line_h     = int(_scale(img, 22))
    pad        = int(_scale(img, 10))
    panel_w    = int(_scale(img, 380))
    panel_h    = pad * 2 + line_h * len(lines)

    panel_h = min(panel_h, h - 10)
    panel_w = min(panel_w, w - 10)

    overlay = img.copy()
    cv2.rectangle(overlay, (5, 5), (5 + panel_w, 5 + panel_h), COLOR_PANEL_BG, -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
    cv2.rectangle(img, (5, 5), (5 + panel_w, 5 + panel_h), (80, 80, 80), 1)

    for idx, line in enumerate(lines):
        y = 5 + pad + line_h * idx + int(_scale(img, 14))
        if y > 5 + panel_h - pad:
            break
        color = (220, 220, 220) if idx > 0 else (0, 220, 220)
        cv2.putText(img, line, (10, y), FONT, font_scale, color, thickness, cv2.LINE_AA)


def draw_results(image_path: str, animals: list, output_path: str):
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Cannot read image: {image_path}")

    # ── Masks ──────────────────────────────────────────────────────────────
    overlay = img.copy()
    for i, animal in enumerate(animals):
        overlay[animal["mask"] == 1] = PALETTE[i % len(PALETTE)]
    cv2.addWeighted(overlay, 0.30, img, 0.70, 0, img)

    th = _th(img)

    # ── Per-animal ─────────────────────────────────────────────────────────
    for i, animal in enumerate(animals):
        color = PALETTE[i % len(PALETTE)]

        contours, _ = cv2.findContours(
            animal["mask"], cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(img, contours, -1, color, th, cv2.LINE_AA)

        x1, y1, _, _ = animal["bbox"]
        label = f"[{i}] {animal['class_name']} {animal['confidence']:.2f}"
        _put_text_with_bg(img, label,
                          (x1, max(y1 - int(_scale(img, 8)), 18)),
                          _fs(img, 0.55), color)

        eyes = animal.get("eyes")
        if not eyes:
            continue

        r_outer = max(5, int(_scale(img, 7)))
        r_inner = max(3, int(_scale(img, 5)))
        for pt in [eyes.left, eyes.right]:
            if pt:
                px, py = int(pt[0]), int(pt[1])
                cv2.circle(img, (px, py), r_outer, (255, 255, 255), -1)
                cv2.circle(img, (px, py), r_inner, COLOR_EYE, -1)

        if eyes.left and eyes.right:
            lp  = (int(eyes.left[0]),  int(eyes.left[1]))
            rp  = (int(eyes.right[0]), int(eyes.right[1]))
            cv2.line(img, lp, rp, COLOR_INTER_EYE, th, cv2.LINE_AA)
            mid = ((lp[0] + rp[0]) // 2,
                   (lp[1] + rp[1]) // 2 - int(_scale(img, 10)))
            d   = animal.get("inter_eye_dist") or 0
            _put_text_with_bg(img, f"{d:.1f}px", mid, _fs(img, 0.45), COLOR_INTER_EYE)

    # ── Cross-animal right-eye lines ───────────────────────────────────────
    for i in range(len(animals)):
        for j in range(i + 1, len(animals)):
            ea = animals[i].get("eyes")
            eb = animals[j].get("eyes")
            if ea and ea.right and eb and eb.right:
                pa  = (int(ea.right[0]), int(ea.right[1]))
                pb  = (int(eb.right[0]), int(eb.right[1]))
                cv2.line(img, pa, pb, COLOR_CROSS, th, cv2.LINE_AA)
                mid = ((pa[0] + pb[0]) // 2,
                       (pa[1] + pb[1]) // 2 - int(_scale(img, 10)))
                d   = animals[i].get(f"cross_{j}_dist") or 0
                _put_text_with_bg(img, f"[{i}]<->[{j}] {d:.1f}px",
                                  mid, _fs(img, 0.45), COLOR_CROSS)

    # ── Info panel ─────────────────────────────────────────────────────────
    _draw_info_panel(img, animals)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(output_path, img)
