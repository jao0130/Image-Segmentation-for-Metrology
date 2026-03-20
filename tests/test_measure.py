import pytest
from src.measure import euclidean_distance, inter_eye_distance, right_eye_distance


def test_horizontal_distance():
    assert euclidean_distance((0, 0), (3, 0)) == pytest.approx(3.0)


def test_vertical_distance():
    assert euclidean_distance((0, 0), (0, 4)) == pytest.approx(4.0)


def test_diagonal_345():
    assert euclidean_distance((0, 0), (3, 4)) == pytest.approx(5.0)


def test_zero_distance():
    assert euclidean_distance((5, 5), (5, 5)) == pytest.approx(0.0)


def test_inter_eye_distance_horizontal():
    assert inter_eye_distance((10, 20), (50, 20)) == pytest.approx(40.0)


def test_right_eye_distance_two_animals():
    assert right_eye_distance((100, 100), (200, 100)) == pytest.approx(100.0)


def test_float_coordinates():
    assert euclidean_distance((1.5, 2.5), (4.5, 6.5)) == pytest.approx(5.0)
