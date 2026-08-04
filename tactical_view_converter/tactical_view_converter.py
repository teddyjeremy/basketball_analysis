import cv2
import numpy as np


class TacticalViewConverter:
    def __init__(self, court_points, output_size=(800, 1200)):
        self.court_points = np.asarray(court_points, dtype=np.float32)
        self.output_width, self.output_height = output_size
        self.matrix = None
        self.inverse_matrix = None

    def fit(self, image_points, valid_mask=None):
        image_points = np.asarray(image_points, dtype=np.float32)
        if valid_mask is None:
            valid_mask = np.ones(len(image_points), dtype=bool)
        valid_mask = np.asarray(valid_mask, dtype=bool)
        if len(image_points) != len(self.court_points):
            raise ValueError("Point sets must have the same length")
        if valid_mask.sum() < 4:
            raise ValueError("At least four valid court points are required")
        self.matrix, _ = cv2.findHomography(
            image_points[valid_mask],
            self.court_points[valid_mask],
            cv2.RANSAC,
            5.0,
        )
        if self.matrix is None:
            raise ValueError("Unable to estimate court homography")
        self.inverse_matrix = np.linalg.inv(self.matrix)
        return self.matrix

    def transform_points(self, points):
        if self.matrix is None:
            raise RuntimeError("Converter must be fitted before transforming points")
        points = np.asarray(points, dtype=np.float32).reshape(-1, 1, 2)
        transformed = cv2.perspectiveTransform(points, self.matrix)
        return transformed.reshape(-1, 2)

    def transform_bbox_feet(self, bbox):
        x1, _, x2, y2 = bbox
        point = np.asarray([[(x1 + x2) / 2.0, y2]], dtype=np.float32)
        return self.transform_points(point)[0]

    def warp_frame(self, frame):
        if self.matrix is None:
            raise RuntimeError("Converter must be fitted before warping frames")
        return cv2.warpPerspective(
            frame,
            self.matrix,
            (self.output_width, self.output_height),
        )

    def draw_court(self, frame, lines, thickness=2):
        output = frame.copy()
        for line in lines:
            if len(line) != 2:
                continue
            start, end = line
            cv2.line(
                output,
                tuple(np.asarray(start, dtype=int)),
                tuple(np.asarray(end, dtype=int)),
                (255, 255, 255),
                thickness,
            )
        return output
