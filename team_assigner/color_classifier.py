import cv2
import numpy as np


class ColorClassifier:
    def __init__(self, crop_size=(64, 96)):
        self.crop_size = crop_size

    def extract(self, frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        crop = crop[: max(1, crop.shape[0] // 2)]
        return np.median(crop.reshape(-1, 3), axis=0)

    def distance(self, color_a, color_b):
        if color_a is None or color_b is None:
            return float("inf")
        return float(np.linalg.norm(np.asarray(color_a) - np.asarray(color_b)))
