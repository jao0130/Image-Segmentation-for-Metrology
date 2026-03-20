from src.filter_images import get_images_with_multiple_animals, ANIMAL_CATEGORY_IDS


def _make_anns(pairs):
    """Helper: [(image_id, category_id), ...] → COCO-like annotations dict."""
    return {
        "annotations": [
            {"id": i, "image_id": img_id, "category_id": cat_id}
            for i, (img_id, cat_id) in enumerate(pairs)
        ]
    }


def test_image_with_two_animals_is_included():
    anns = _make_anns([(1, 17), (1, 18)])  # cat + dog in image 1
    assert 1 in get_images_with_multiple_animals(anns)


def test_image_with_one_animal_is_excluded():
    anns = _make_anns([(2, 17)])
    assert 2 not in get_images_with_multiple_animals(anns)


def test_non_animal_categories_are_ignored():
    # person(1) + bicycle(2) + one cat(17) = only 1 animal
    anns = _make_anns([(3, 1), (3, 2), (3, 17)])
    assert 3 not in get_images_with_multiple_animals(anns)


def test_min_count_three():
    anns = _make_anns([(4, 17), (4, 18), (4, 19), (5, 17), (5, 18)])
    result = get_images_with_multiple_animals(anns, min_count=3)
    assert 4 in result
    assert 5 not in result


def test_empty_annotations_returns_empty():
    assert get_images_with_multiple_animals({"annotations": []}) == []


def test_animal_ids_are_correct():
    # Spot-check a few known IDs
    assert 17 in ANIMAL_CATEGORY_IDS  # cat
    assert 25 in ANIMAL_CATEGORY_IDS  # giraffe
    assert 1 not in ANIMAL_CATEGORY_IDS  # person
