from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


class CourtKeypointDetector:
    def __init__(self, model_path, confidence=0.25, device=None):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Court keypoint model not found: {model_path}")
        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.device = device

    def detect(self, frame):
        result = self.model.predict(
            source=frame,
            conf=self.confidence,
            device=self.device,
            verbose=False,
        )[0]
        if result.keypoints is None:
            return np.empty((0, 2), dtype=np.float32), np.empty((0,), dtype=np.float32)
        points = result.keypoints.xy[0].detach().cpu().numpy().astype(np.float32)
        if result.keypoints.conf is None:
            confidence = np.ones(len(points), dtype=np.float32)
        else:
            confidence = result.keypoints.conf[0].detach().cpu().numpy().astype(np.float32)
        return points, confidence

    @staticmethod
    def valid_keypoints(points, confidence, threshold=0.5):
        if len(points) != len(confidence):
            raise ValueError("Keypoint and confidence lengths must match")
        return confidence >= threshold

    @staticmethod
    def refine_points(frame, points, window=7):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        refined = []
        for point in points:
            x, y = point
            x1 = max(0, int(x) - window)
            y1 = max(0, int(y) - window)
            x2 = min(gray.shape[1], int(x) + window + 1)
            y2 = min(gray.shape[0], int(y) + window + 1)
            patch = gray[y1:y2, x1:x2]
            if patch.size == 0:
                refined.append(point)
                continue
            _, _, _, max_location = cv2.minMaxLoc(patch)
            refined.append([x1 + max_location[0], y1 + max_location[1]])
        return np.asarray(refined, dtype=np.float32)
