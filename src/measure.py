import math


def euclidean_distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return math.sqrt((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2)


def inter_eye_distance(left_eye: tuple, right_eye: tuple) -> float:
    """Pixel distance between left and right eye of a single animal."""
    return euclidean_distance(left_eye, right_eye)


def right_eye_distance(right_eye_a: tuple, right_eye_b: tuple) -> float:
    """Pixel distance between right eyes of two different animals."""
    return euclidean_distance(right_eye_a, right_eye_b)
