import json
import os
import random
import sys
from pathlib import Path

import cv2
import pandas as pd
from dotenv import load_dotenv

# Project root and src path (independent of working directory)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))

load_dotenv(PROJECT_ROOT / ".env")

from download_coco import download_coco
from eye_keypoint_model.predict import load_pose_model, detect_eyes_pose
from filter_images import get_images_with_multiple_animals
from measure import inter_eye_distance, right_eye_distance
from segment import load_model, segment_animals
from visualize import draw_results

DATA_DIR = os.getenv("DATA_DIR", str(PROJECT_ROOT / "data/coco"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR", str(PROJECT_ROOT / "output"))
SAMPLE_COUNT = int(os.getenv("SAMPLE_COUNT", "100"))
RANDOM_SEED = int(os.getenv("RANDOM_SEED", "42"))


def run():
    pose_model = load_pose_model() 

    paths = download_coco(DATA_DIR)
    images_dir = paths["images_dir"]

    with open(paths["annotations_path"], encoding="utf-8") as f:
        annotations = json.load(f)

    valid_ids = get_images_with_multiple_animals(annotations)
    random.seed(RANDOM_SEED)
    sample_ids = random.sample(valid_ids, min(SAMPLE_COUNT, len(valid_ids)))
    id_to_file = {img["id"]: img["file_name"] for img in annotations["images"]}
    print(f"Found {len(valid_ids)} images with 2+ animals. Processing {len(sample_ids)}.")

    model = load_model()
    Path(f"{OUTPUT_DIR}/images").mkdir(parents=True, exist_ok=True)

    rows = []

    for img_id in sample_ids:
        img_path = f"{images_dir}/{id_to_file[img_id]}"
        if not Path(img_path).exists():
            print(f"  Skip {img_id}: file not found")
            continue

        animals = segment_animals(model, img_path)
        if len(animals) < 2:
            print(f"  Skip {img_id}: YOLOv8 found {len(animals)} animal(s)")
            continue

        image = cv2.imread(img_path)

        for animal in animals:
            eyes = detect_eyes_pose(pose_model, image, animal["bbox"])
            animal["eyes"] = eyes
            if eyes.left and eyes.right:
                animal["inter_eye_dist"] = inter_eye_distance(eyes.left, eyes.right)
            else:
                animal["inter_eye_dist"] = None

        for i in range(len(animals)):
            for j in range(i + 1, len(animals)):
                ea = animals[i]["eyes"]
                eb = animals[j]["eyes"]
                if ea.right and eb.right:
                    d = right_eye_distance(ea.right, eb.right)
                    animals[i][f"cross_{j}_dist"] = d
                    rows.append({
                        "image_id": img_id,
                        "type": "cross_animal_right_eye",
                        "animal_a": i,
                        "animal_b": j,
                        "class_a": animals[i]["class_name"],
                        "class_b": animals[j]["class_name"],
                        "left_eye_x": None,
                        "left_eye_y": None,
                        "right_eye_x": None,
                        "right_eye_y": None,
                        "distance_px": round(d, 2),
                        "eye_method": None,
                    })

        for i, a in enumerate(animals):
            rows.append({
                "image_id": img_id,
                "type": "inter_eye",
                "animal_a": i,
                "animal_b": None,
                "class_a": a["class_name"],
                "class_b": None,
                "left_eye_x": a["eyes"].left[0] if a["eyes"].left else None,
                "left_eye_y": a["eyes"].left[1] if a["eyes"].left else None,
                "right_eye_x": a["eyes"].right[0] if a["eyes"].right else None,
                "right_eye_y": a["eyes"].right[1] if a["eyes"].right else None,
                "distance_px": round(a["inter_eye_dist"], 2) if a["inter_eye_dist"] else None,
                "eye_method": a["eyes"].method,
            })

        out_path = f"{OUTPUT_DIR}/images/{img_id}_result.jpg"
        draw_results(img_path, animals, out_path)
        print(f"  [{img_id}] {len(animals)} animals → {out_path}")

    df = pd.DataFrame(rows)
    csv_path = f"{OUTPUT_DIR}/measurements.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nDone! {len(rows)} measurement rows saved to {csv_path}")


if __name__ == "__main__":
    run()
