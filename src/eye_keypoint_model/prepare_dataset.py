"""
Prepare AP-10K dataset for YOLOv8-pose training.

AP-10K is a large-scale animal pose estimation benchmark.
Source: https://github.com/AlexTheBad/AP-10K

Keypoints (17 total, same layout as COCO-person):
  0: left_eye      1: right_eye     2: nose
  3: neck          4: root_of_tail
  5: left_shoulder  6: left_elbow    7: left_front_paw
  8: right_shoulder 9: right_elbow  10: right_front_paw
  11: left_hip     12: left_knee    13: left_back_paw
  14: right_hip    15: right_knee   16: right_back_paw

This script converts to YOLOv8 pose format keeping ONLY
left_eye (0) and right_eye (1) to minimise annotation noise.

Usage:
    python src/keypoint_model/prepare_dataset.py

Expected input structure:
    data/ap10k/
        annotations/
            ap10k-train-split1.json
            ap10k-val-split1.json
        data/           <- images
"""

import json
import shutil
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Indices of the two eye keypoints within AP-10K's 17-kp schema
EYE_KP_INDICES = [0, 1, 2]   # left_eye, right_eye, nose

# Minimum eye visibility: at least one eye must be visible (v=2) or labelled (v=1)
MIN_EYE_VISIBILITY = 1


def prepare_dataset(
    ap10k_dir: str = str(PROJECT_ROOT / "data/ap10k"),
    output_dir: str = str(PROJECT_ROOT / "data/ap10k_eye_keypoints"),
) -> str | None:
    """
    Convert AP-10K COCO-format annotations to YOLOv8 pose format.

    Returns path to the generated dataset.yaml, or None if ap10k_dir missing.
    """
    ap10k = Path(ap10k_dir)
    out = Path(output_dir)

    if not ap10k.exists():
        print("AP-10K dataset not found.")
        print("Download from: https://github.com/AlexTheBad/AP-10K")
        print(f"Expected location: {ap10k.resolve()}/")
        print("\nExpected structure:")
        print("  data/ap10k/annotations/ap10k-train-split1.json")
        print("  data/ap10k/annotations/ap10k-val-split1.json")
        print("  data/ap10k/data/  (images)")
        return None

    for split in ("train", "val"):
        ann_path = ap10k / f"annotations/ap10k-{split}-split1.json"
        img_dir = ap10k / "data"
        if not ann_path.exists():
            print(f"  Missing annotation file: {ann_path}")
            continue

        out_img = out / "images" / split
        out_lbl = out / "labels" / split
        out_img.mkdir(parents=True, exist_ok=True)
        out_lbl.mkdir(parents=True, exist_ok=True)

        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        n = _convert_split(coco, img_dir, out_img, out_lbl)
        print(f"  {split}: {n} instances written")

    yaml_path = _write_yaml(out)
    print(f"\nDataset ready: {yaml_path}")
    return str(yaml_path)


def _convert_split(
    coco: dict,
    img_dir: Path,
    out_img: Path,
    out_lbl: Path,
) -> int:
    id_to_img = {img["id"]: img for img in coco["images"]}

    # Group annotations by image_id
    img_anns: dict[int, list] = {}
    for ann in coco["annotations"]:
        img_anns.setdefault(ann["image_id"], []).append(ann)

    n_written = 0

    for img_id, anns in img_anns.items():
        img_info = id_to_img[img_id]
        iw, ih = img_info["width"], img_info["height"]
        img_file = img_info["file_name"]

        lines = []
        for ann in anns:
            kps = ann.get("keypoints", [])
            if len(kps) < 3 * 17:
                continue  # incomplete annotation

            # Require at least one eye to be labelled
            lv = kps[0 * 3 + 2]  # left_eye visibility
            rv = kps[1 * 3 + 2]  # right_eye visibility
            if lv < MIN_EYE_VISIBILITY and rv < MIN_EYE_VISIBILITY:
                continue

            # BBox: COCO stores [x, y, w, h]
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / iw
            cy = (by + bh / 2) / ih
            nw = bw / iw
            nh = bh / ih

            # Two eye keypoints only
            kp_parts = []
            for ki in EYE_KP_INDICES:
                kx = kps[ki * 3] / iw
                ky = kps[ki * 3 + 1] / ih
                kv = kps[ki * 3 + 2]
                kp_parts.append(f"{kx:.6f} {ky:.6f} {int(kv)}")

            line = f"0 {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f} " + " ".join(kp_parts)
            lines.append(line)

        if not lines:
            continue

        # Copy image with CLAHE — matches inference preprocessing in predict.py
        src = img_dir / img_file
        dst = out_img / img_file
        if src.exists() and not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            img = cv2.imread(str(src))
            if img is not None:
                cv2.imwrite(str(dst), _clahe(img))
            else:
                shutil.copy2(src, dst)

        lbl_path = out_lbl / (Path(img_file).stem + ".txt")
        lbl_path.write_text("\n".join(lines), encoding="utf-8")
        n_written += len(lines)

    return n_written


def _write_yaml(out: Path) -> Path:
    content = f"""\
# Animal eye keypoint dataset (converted from AP-10K)
path: {out.resolve().as_posix()}
train: images/train
val:   images/val

# 2 keypoints: [left_eye, right_eye]
kpt_shape: [3, 3]   # [num_keypoints, (x, y, visibility)]
flip_idx: [1, 0, 2]    # horizontal flip: left_eye <-> right_eye

names:
  0: animal
"""
    yaml_path = out / "dataset.yaml"
    yaml_path.write_text(content, encoding="utf-8")
    return yaml_path


def _clahe(img: np.ndarray) -> np.ndarray:
    """Apply CLAHE on the L channel (LAB) — matches predict.py inference."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4)).apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


if __name__ == "__main__":
    prepare_dataset()
