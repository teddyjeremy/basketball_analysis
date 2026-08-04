from pathlib import Path

import cv2

from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from utils.video_writer import VideoWriter


class VideoAnalysis:
    def __init__(self, player_tracker, ball_tracker, frame_stride=1):
        if frame_stride < 1:
            raise ValueError("frame_stride must be at least 1")
        self.player_tracker = player_tracker
        self.ball_tracker = ball_tracker
        self.frame_stride = frame_stride

    def read_frames(self, video_path):
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise FileNotFoundError(f"Unable to open video: {video_path}")

        frames = []
        while True:
            success, frame = capture.read()
            if not success:
                break
            frames.append(frame)
        capture.release()
        return frames

    def run_detection(self, video_path):
        frames = self.read_frames(video_path)
        selected_frames = frames[::self.frame_stride]
        player_tracks = self.player_tracker.get_object_tracks(selected_frames)
        ball_tracks = self.ball_tracker.get_best_detections(selected_frames)
        return {
            "frames": selected_frames,
            "player_tracks": player_tracks,
            "ball_tracks": ball_tracks,
        }

    def write_frames(self, frames, output_path, fps):
        if not frames:
            raise ValueError("No frames available for writing")
        height, width = frames[0].shape[:2]
        with VideoWriter(output_path, fps, (width, height)) as writer:
            for frame in frames:
                writer.write(frame)
