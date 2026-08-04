import argparse
from pathlib import Path

from trackers.player_tracker import PlayerTracker
from trackers.ball_tracker import BallTracker
from pipelines.video_analysis import VideoAnalysis
from utils.config_loader import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--input", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    input_path = args.input or config["video"]["input"]
    output_path = args.output or config["video"]["output"]

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

    pipeline = VideoAnalysis(
        player_tracker=player_tracker,
        ball_tracker=ball_tracker,
        frame_stride=config["video"]["stride"],
    )
    result = pipeline.run_detection(input_path)

    capture_fps = 30.0
    pipeline.write_frames(
        result["frames"],
        Path(output_path),
        capture_fps / config["video"]["stride"],
    )


if __name__ == "__main__":
    main()
