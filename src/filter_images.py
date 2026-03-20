# COCO annotation category IDs for animals (1-indexed in instances JSON)
ANIMAL_CATEGORY_IDS = {16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
# bird=16, cat=17, dog=18, horse=19, sheep=20, cow=21, elephant=22, bear=23, zebra=24, giraffe=25


def get_images_with_multiple_animals(annotations: dict, min_count: int = 2) -> list:
    """Return image IDs containing at least min_count animal instances."""
    counts = {}
    for ann in annotations["annotations"]:
        if ann["category_id"] in ANIMAL_CATEGORY_IDS:
            counts[ann["image_id"]] = counts.get(ann["image_id"], 0) + 1
    return [img_id for img_id, n in counts.items() if n >= min_count]
