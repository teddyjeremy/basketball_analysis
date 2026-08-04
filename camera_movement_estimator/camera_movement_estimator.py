from pathlib import Path

import cv2
import numpy as np


class CameraMovementEstimator:
    def __init__(
        self,
        minimum_distance=5.0,
        quality_level=0.01,
        min_distance=7,
        block_size=7,
        max_corners=300,
        search_radius=30,
        smoothing_window=7,
    ):
        self.minimum_distance = minimum_distance
        self.quality_level = quality_level
        self.min_distance = min_distance
        self.block_size = block_size
        self.max_corners = max_corners
        self.search_radius = search_radius
        self.smoothing_window = smoothing_window
        self.previous_gray = None

    def _feature_mask(self, frame):
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        height, width = mask.shape
        margin_y = int(height * 0.08)
        mask[margin_y:height - margin_y, :] = 255
        return mask

    def _detect_features(self, gray):
        return cv2.goodFeaturesToTrack(
            gray,
            mask=self._feature_mask(gray),
            maxCorners=self.max_corners,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            blockSize=self.block_size,
        )

    def _estimate_translation(self, previous_gray, current_gray):
        previous_points = self._detect_features(previous_gray)
        if previous_points is None or len(previous_points) < 4:
            return np.zeros(2, dtype=np.float32), 0

        current_points, status, _ = cv2.calcOpticalFlowPyrLK(
            previous_gray,
            current_gray,
            previous_points,
            None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                30,
                0.01,
            ),
        )

        if current_points is None or status is None:
            return np.zeros(2, dtype=np.float32), 0

        valid_previous = previous_points[status.ravel() == 1]
        valid_current = current_points[status.ravel() == 1]

        if len(valid_previous) < 4:
            return np.zeros(2, dtype=np.float32), len(valid_previous)

        displacement = valid_current - valid_previous
        median_displacement = np.median(displacement, axis=0)
        residual = np.linalg.norm(displacement - median_displacement, axis=1)
        inliers = residual <= max(self.minimum_distance, 3 * np.median(residual))

        if inliers.sum() < 4:
            return median_displacement.astype(np.float32), int(inliers.sum())

        translation = np.median(displacement[inliers], axis=0)
        return translation.astype(np.float32), int(inliers.sum())

    def estimate(self, frames):
        if not frames:
            return np.empty((0, 2), dtype=np.float32)

        gray_frames = [cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) for frame in frames]
        movements = np.zeros((len(frames), 2), dtype=np.float32)

        for index in range(1, len(gray_frames)):
            translation, _ = self._estimate_translation(
                gray_frames[index - 1],
                gray_frames[index],
            )
            movements[index] = translation

        if self.smoothing_window > 1:
            kernel_size = self.smoothing_window
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            for axis in range(2):
                movements[:, axis] = np.convolve(
                    movements[:, axis],
                    kernel,
                    mode="same",
                )

        return movements

    def get_camera_movement(self, frames):
        return self.estimate(frames)

    def accumulate(self, frame_movements):
        frame_movements = np.asarray(frame_movements, dtype=np.float32)
        if frame_movements.size == 0:
            return np.empty((0, 2), dtype=np.float32)
        return np.cumsum(frame_movements, axis=0)

    def save(self, movements, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.save(path, np.asarray(movements, dtype=np.float32))

    def load(self, path):
        return np.load(path).astype(np.float32)
