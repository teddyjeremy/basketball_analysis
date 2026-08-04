from utils.bbox_utils import get_distance, get_foot_position


class BallAcquisition:
    def __init__(self, max_distance=120.0, max_gap=8):
        self.max_distance = max_distance
        self.max_gap = max_gap

    def _player_point(self, detection):
        return get_foot_position(detection["box"])

    def _ball_point(self, detection):
        box = detection["box"]
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def assign(self, player_tracks, ball_detection):
        if not ball_detection:
            return None

        ball_point = self._ball_point(ball_detection)
        candidates = []

        for player_id, detection in player_tracks.items():
            distance = get_distance(
                self._player_point(detection),
                ball_point,
            )
            if distance <= self.max_distance:
                candidates.append((distance, player_id))

        if not candidates:
            return None

        candidates.sort(key=lambda item: item[0])
        return candidates[0][1]

    def assign_sequence(self, player_tracks, ball_detections):
        possession = []
        previous_player = None
        missing = 0

        for tracks, ball_detection in zip(player_tracks, ball_detections):
            player_id = self.assign(tracks, ball_detection)
            if player_id is None:
                missing += 1
                if missing <= self.max_gap:
                    player_id = previous_player
            else:
                missing = 0
                previous_player = player_id
            possession.append(player_id)

        return possession
