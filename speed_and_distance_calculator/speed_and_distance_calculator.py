import numpy as np


class SpeedAndDistanceCalculator:
    def __init__(self, fps, pixels_per_meter=10.0, smoothing_window=5):
        if fps <= 0:
            raise ValueError("fps must be positive")
        if pixels_per_meter <= 0:
            raise ValueError("pixels_per_meter must be positive")
        self.fps = float(fps)
        self.pixels_per_meter = float(pixels_per_meter)
        self.smoothing_window = max(1, int(smoothing_window))

    def calculate_distance(self, positions, camera_movement=None):
        distances = np.zeros(len(positions), dtype=np.float32)
        for index in range(1, len(positions)):
            current = positions[index]
            previous = positions[index - 1]
            if current is None or previous is None:
                continue

            displacement = np.asarray(current, dtype=np.float32) - np.asarray(previous, dtype=np.float32)
            if camera_movement is not None and index < len(camera_movement):
                displacement -= np.asarray(camera_movement[index], dtype=np.float32)
            distances[index] = np.linalg.norm(displacement) / self.pixels_per_meter
        return distances

    def calculate_speed(self, distances):
        speed = np.asarray(distances, dtype=np.float32) * self.fps
        if self.smoothing_window > 1 and len(speed) >= self.smoothing_window:
            kernel = np.ones(self.smoothing_window, dtype=np.float32) / self.smoothing_window
            speed = np.convolve(speed, kernel, mode="same")
        return speed

    def calculate_track_metrics(self, positions, camera_movement=None):
        distance = self.calculate_distance(positions, camera_movement)
        speed = self.calculate_speed(distance)
        return {
            "distance_per_frame": distance,
            "speed_mps": speed,
            "total_distance_m": float(distance.sum()),
            "max_speed_mps": float(speed.max()) if len(speed) else 0.0,
            "mean_speed_mps": float(speed.mean()) if len(speed) else 0.0,
        }
