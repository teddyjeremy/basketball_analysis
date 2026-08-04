import argparse
from pathlib import Path

import cv2

from analysis.report import build_report
from ball_acquisition.ball_acquisition import BallAcquisition
from camera_movement_estimator.camera_movement_estimator import CameraMovementEstimator
from drawers.ball_drawer import BallDrawer
from drawers.player_drawer import PlayerDrawer
from drawers.possession_drawer import PossessionDrawer
from pass_and_interception_detector.pass_and_interception_detector import (
    PassAndInterceptionDetector,
)
from speed_and_distance_calculator.speed_and_distance_calculator import (
    SpeedAndDistanceCalculator,
)
from team_assigner.team_assigner import TeamAssigner
from trackers.ball_tracker import BallTracker
from trackers.player_tracker import PlayerTracker
from utils.config_loader import load_config
from utils.serialization import save_json
from utils.video_writer import VideoWriter


def read_video(path, stride):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise FileNotFoundError(f"Unable to open video: {path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0

    frames = []
    frame_index = 0
    while True:
        success, frame = capture.read()
        if not success:
            break
        if frame_index % stride == 0:
            frames.append(frame)
        frame_index += 1

    capture.release()
    if not frames:
        raise ValueError(f"No frames were read from video: {path}")
    return frames, fps / stride


def build_team_assignments(frames, player_tracks, config):
    assigner = TeamAssigner(
        n_teams=config["team_assignment"]["teams"],
        random_state=config["team_assignment"]["random_state"],
    )

    first_tracks = player_tracks[0]
    if len(first_tracks) < config["team_assignment"]["teams"]:
        raise ValueError("Not enough players in the first frame to assign teams")

    assigner.fit(frames[0], first_tracks)
    assignments = []
    stable_assignments = dict(assigner.player_team)

    for frame, tracks in zip(frames, player_tracks):
        frame_assignments = assigner.update_from_tracks(frame, tracks)
        stable_assignments.update(
            {
                player_id: team_id
                for player_id, team_id in frame_assignments.items()
                if team_id is not None
            }
        )
        assignments.append(frame_assignments)

    return assignments, stable_assignments


def draw_analysis(frames, player_tracks, ball_tracks, possession, team_assignments, output_path, fps):
    player_drawer = PlayerDrawer()
    ball_drawer = BallDrawer()
    possession_drawer = PossessionDrawer()

    height, width = frames[0].shape[:2]
    with VideoWriter(output_path, fps, (width, height)) as writer:
        for frame, players, ball, player_id, teams in zip(
            frames,
            player_tracks,
            ball_tracks,
            possession,
            team_assignments,
        ):
            output = player_drawer.draw(frame, players, teams)
            output = ball_drawer.draw(output, ball)
            output = possession_drawer.draw(output, players, player_id)
            writer.write(output)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--metrics", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = Path(args.input or config["video"]["input"])
    output_path = Path(args.output or config["video"]["output"])
    metrics_path = Path(args.metrics or config["outputs"]["metrics"])
    stride = max(1, int(config["video"]["stride"]))

    frames, fps = read_video(input_path, stride)
    detection = config["detection"]

    player_tracker = PlayerTracker(
        model_path=config["models"]["player"],
        confidence=detection["player_confidence"],
        iou=detection["iou"],
        batch_size=detection["batch_size"],
        device=detection["device"],
    )
    ball_tracker = BallTracker(
        model_path=config["models"]["ball"],
        confidence=detection["ball_confidence"],
        iou=detection["iou"],
        batch_size=detection["batch_size"],
        device=detection["device"],
    )

    player_tracks = player_tracker.get_object_tracks(frames)
    ball_tracks = ball_tracker.get_best_detections(frames)

    team_assignments, stable_team_assignments = build_team_assignments(
        frames,
        player_tracks,
        config,
    )

    possession_tracker = BallAcquisition(
        max_distance=config["possession"]["max_distance"],
        max_gap=config["possession"]["max_gap"],
    )
    possession = possession_tracker.assign_sequence(player_tracks, ball_tracks)

    event_detector = PassAndInterceptionDetector()
    events = event_detector.detect(possession, stable_team_assignments)

    camera_estimator = CameraMovementEstimator()
    camera_movement = camera_estimator.get_camera_movement(frames)

    motion_calculator = SpeedAndDistanceCalculator(
        fps=fps,
        pixels_per_meter=config["motion"]["pixels_per_meter"],
        smoothing_window=config["motion"]["smoothing_window"],
    )

    player_positions = {}
    player_metrics = {}
    for player_id in stable_team_assignments:
        positions = []
        for tracks in player_tracks:
            detection_item = tracks.get(player_id)
            if detection_item is None:
                positions.append(None)
                continue
            x1, y1, x2, y2 = detection_item["box"]
            positions.append(((x1 + x2) / 2.0, y2))
        player_positions[player_id] = positions
        player_metrics[player_id] = motion_calculator.calculate_track_metrics(
            positions,
            camera_movement=camera_movement,
        )

    report = build_report(
        possession=possession,
        events=events,
        player_metrics=player_metrics,
        team_assignments=stable_team_assignments,
    )
    report["video"] = {
        "input": str(input_path),
        "output": str(output_path),
        "fps": fps,
        "frames": len(frames),
        "stride": stride,
    }

    draw_analysis(
        frames,
        player_tracks,
        ball_tracks,
        possession,
        team_assignments,
        output_path,
        fps,
    )
    save_json(report, metrics_path)


if __name__ == "__main__":
    main()
