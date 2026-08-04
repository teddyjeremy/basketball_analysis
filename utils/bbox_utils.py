import numpy as np


def get_center_of_bbox(bbox):
    x1, y1, x2, y2 = bbox
    return int((x1 + x2) / 2), int((y1 + y2) / 2)


def get_bbox_width(bbox):
    x1, _, x2, _ = bbox
    return int(x2 - x1)


def get_bbox_height(bbox):
    _, y1, _, y2 = bbox
    return int(y2 - y1)


def get_foot_position(bbox):
    x1, _, x2, y2 = bbox
    return int((x1 + x2) / 2), int(y2)


def get_bbox_area(bbox):
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def get_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    union = get_bbox_area(box_a) + get_bbox_area(box_b) - intersection

    return intersection / union if union > 0 else 0.0


def get_distance(point_a, point_b):
    return float(np.linalg.norm(np.asarray(point_a) - np.asarray(point_b)))
