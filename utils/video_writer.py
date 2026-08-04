from pathlib import Path

import cv2


class VideoWriter:
    def __init__(self, output_path, fps, frame_size, codec="mp4v"):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if fps <= 0:
            raise ValueError("fps must be positive")
        if len(frame_size) != 2:
            raise ValueError("frame_size must contain width and height")
        self.output_path = output_path
        self.writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            float(fps),
            tuple(map(int, frame_size)),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Unable to open video writer: {output_path}")

    def write(self, frame):
        self.writer.write(frame)

    def release(self):
        self.writer.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
