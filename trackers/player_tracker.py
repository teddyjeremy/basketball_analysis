from pathlib import Path

import supervision as sv
from ultralytics import YOLO


class PlayerTracker:
    def __init__(
        self,
        model_path,
        confidence=0.35,
        iou=0.5,
        batch_size=8,
        device=None,
        player_class_name="Player",
    ):
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Player model not found: {model_path}")

        self.model = YOLO(str(model_path))
        self.confidence = confidence
        self.iou = iou
        self.batch_size = batch_size
        self.device = device
        self.player_class_name = player_class_name
        self.tracker = sv.ByteTrack()

    def detect_frames(self, frames):
        if not frames:
            return []

        results = []
        for start in range(0, len(frames), self.batch_size):
            batch = frames[start:start + self.batch_size]
            batch_results = self.model.predict(
                source=batch,
                conf=self.confidence,
                iou=self.iou,
                device=self.device,
                verbose=False,
            )
            results.extend(batch_results)
        return results

    def get_object_tracks(self, frames):
        detections = self.detect_frames(frames)
        tracks = []
        self.tracker.reset()

        names = self.model.names
        player_class_ids = {
            class_id
            for class_id, class_name in names.items()
            if class_name.lower() == self.player_class_name.lower()
        }

        if not player_class_ids:
            raise ValueError(
                f"Class '{self.player_class_name}' was not found in the model classes."
            )

        for detection in detections:
            supervision_detections = sv.Detections.from_ultralytics(detection)
            tracked_detections = self.tracker.update_with_detections(
                supervision_detections
            )

            frame_tracks = {}

            for index in range(len(tracked_detections)):
                class_id = int(tracked_detections.class_id[index])
                tracker_id = tracked_detections.tracker_id[index]

                if tracker_id is None or class_id not in player_class_ids:
                    continue

                x1, y1, x2, y2 = tracked_detections.xyxy[index].tolist()
                confidence = float(tracked_detections.confidence[index])

                frame_tracks[int(tracker_id)] = {
                    "box": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                }

            tracks.append(frame_tracks)

        return tracks
