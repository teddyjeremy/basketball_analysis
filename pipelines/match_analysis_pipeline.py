from collections import defaultdict

import numpy as np

from analysis.report import build_report
from ball_acquisition.ball_acquisition import BallAcquisition
from pass_and_interception_detector.pass_and_interception_detector import PassAndInterceptionDetector
from speed_and_distance_calculator.speed_and_distance_calculator import SpeedAndDistanceCalculator
from utils.interpolation import interpolate_missing


class MatchAnalysisPipeline:
    def __init__(self, fps, pixels_per_meter=10.0, possession_distance=120.0):
        self.ball_acquisition = BallAcquisition(max_distance=possession_distance)
        self.event_detector = PassAndInterceptionDetector()
        self.motion = SpeedAndDistanceCalculator(
            fps=fps,
            pixels_per_meter=pixels_per_meter,
        )

    def interpolate_player_positions(self, player_tracks):
        player_ids = sorted({player_id for frame in player_tracks for player_id in frame})
        positions = defaultdict(list)
        for frame in player_tracks:
            for player_id in player_ids:
                detection = frame.get(player_id)
                if detection is None:
                    positions[player_id].append(None)
                else:
                    x1, y1, x2, y2 = detection["box"]
                    positions[player_id].append(((x1 + x2) / 2.0, y2))
        return {
            player_id: interpolate_missing(values)
            for player_id, values in positions.items()
        }

    def calculate_player_metrics(self, player_tracks, camera_movement=None):
        positions = self.interpolate_player_positions(player_tracks)
        metrics = {}
        for player_id, player_positions in positions.items():
            metrics[player_id] = self.motion.calculate_track_metrics(
                player_positions,
                camera_movement=camera_movement,
            )
        return metrics

    def run(self, player_tracks, ball_tracks, team_assignments, camera_movement=None):
        possession = self.ball_acquisition.assign_sequence(
            player_tracks,
            ball_tracks,
        )
        events = self.event_detector.detect(
            possession,
            team_assignments,
        )
        player_metrics = self.calculate_player_metrics(
            player_tracks,
            camera_movement,
        )
        return build_report(
            possession=possession,
            events=events,
            player_metrics=player_metrics,
            team_assignments=team_assignments,
        )
