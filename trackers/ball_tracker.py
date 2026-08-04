from pathlib import Path

from ultralytics import YOLO


class BallTracker:
    def __init__(
        self,
        model_path,
        confidence=0.15,
        iou=0.5,
        batch_size=8,
        device=None,
        ball_class_name="Ball",
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Ball model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.batch_size = batch_size
        self.device = device
        self.ball_class_name = ball_class_name

    def detect_frames(self, frames):
        if not frames:
            return []

        detections = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start:start + self.batch_size]
            detections.extend(
                self.model.predict(
                    source=batch,
                    conf=self.confidence,
                    iou=self.iou,
                    device=self.device,
                    verbose=False,
                )
            )

        return detections

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = []
        names = self.model.names
        ball_class_ids = {
            class_id
            for class_id, class_name in names.items()
            if class_name.lower() == self.ball_class_name.lower()
        }

        if not ball_class_ids:
            raise ValueError(
                f"Class '{self.ball_class_name}' was not found in the model classes."
            )

        for detection in detections:
            frame_tracks = []

            if detection.boxes is None:
                tracks.append(frame_tracks)
                continue

            for index in range(len(detection.boxes)):
                class_id = int(detection.boxes.cls[index].item())
                if class_id not in ball_class_ids:
                    continue

                xyxy = detection.boxes.xyxy[index].tolist()
                confidence = float(detection.boxes.conf[index].item())

                frame_tracks.append(
                    {
                        "box": xyxy,
                        "confidence": confidence,
                        "class_id": class_id,
                    }
                )

            frame_tracks.sort(
                key=lambda item: item["confidence"],
                reverse=True,
            )
            tracks.append(frame_tracks)

        return tracks

    def get_best_detections(self, frames):
        tracks = self.get_object_tracks(frames)
        return [frame_tracks[0] if frame_tracks else None for frame_tracks in tracks]
