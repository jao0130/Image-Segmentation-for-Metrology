import cv2
import numpy as np
from ultralytics import YOLO

# YOLOv8 COCO80 class IDs (0-indexed): bird=14, cat=15, dog=16, horse=17, sheep=18,
# cow=19, elephant=20, bear=21, zebra=22, giraffe=23
ANIMAL_CLASS_IDS = {14, 15, 16, 17, 18, 19, 20, 21, 22, 23}


def load_model(model_path: str = "yolov8n-seg.pt") -> YOLO:
    """Load YOLOv8-seg model (auto-downloads on first run)."""
    return YOLO(model_path)


def segment_animals(model: YOLO, image_path: str, conf: float = 0.15) -> list:
    """
    Run YOLOv8-seg on image, return only animal detections.

    Returns list of dicts:
        {
            "class_id": int,
            "class_name": str,
            "confidence": float,
            "bbox": (x1, y1, x2, y2),  # ints, image coords
            "mask": np.ndarray           # (H, W) binary uint8
        }
    """
    results = model(image_path, conf=conf, verbose=False)[0]
    animals = []

    if results.masks is None:
        return animals

    orig_h, orig_w = results.orig_shape

    for i, box in enumerate(results.boxes):
        cls_id = int(box.cls[0])
        if cls_id not in ANIMAL_CLASS_IDS:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        # results.masks.data[i] corresponds to results.boxes[i] per ultralytics API contract
        mask_raw = results.masks.data[i].cpu().numpy()
        mask = cv2.resize(mask_raw, (orig_w, orig_h),
                          interpolation=cv2.INTER_NEAREST).astype(np.uint8)

        animals.append({
            "class_id": cls_id,
            "class_name": results.names[cls_id],
            "confidence": float(box.conf[0]),
            "bbox": (x1, y1, x2, y2),
            "mask": mask,
        })

    return animals
