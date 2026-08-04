from utils.bbox_utils import get_bbox_area, get_center_of_bbox, get_distance, get_iou


def test_center():
    assert get_center_of_bbox([0, 0, 10, 20]) == (5, 10)


def test_area():
    assert get_bbox_area([0, 0, 10, 20]) == 200


def test_iou():
    assert get_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0


def test_distance():
    assert get_distance((0, 0), (3, 4)) == 5.0
