import numpy as np


def tracking_iou(box_a, box_b):
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - intersection
    return intersection / union if union else 0.0


def mean_position_error(predicted, target):
    predicted = np.asarray(predicted, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    if predicted.shape != target.shape:
        raise ValueError("Predicted and target shapes must match")
    return float(np.linalg.norm(predicted - target, axis=-1).mean())


def possession_percentage(possession):
    valid = [value for value in possession if value is not None]
    if not valid:
        return {}
    unique, counts = np.unique(valid, return_counts=True)
    total = counts.sum()
    return {int(player): float(count / total) for player, count in zip(unique, counts)}
