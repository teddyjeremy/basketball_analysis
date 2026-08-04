from collections import Counter, defaultdict

import cv2
import numpy as np
from sklearn.cluster import KMeans


class TeamAssigner:
    def __init__(self, n_teams=2, random_state=42, crop_size=(64, 96)):
        if n_teams < 2:
            raise ValueError("n_teams must be at least 2")
        self.n_teams = n_teams
        self.random_state = random_state
        self.crop_size = crop_size
        self.kmeans = None
        self.player_team = {}
        self.team_colors = {}

    def _extract_features(self, frame, bbox):
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, width - 1))
        x2 = max(0, min(x2, width))
        y1 = max(0, min(y1, height - 1))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, self.crop_size, interpolation=cv2.INTER_AREA)
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        top = crop[: max(1, crop.shape[0] // 2)]
        features = top.reshape(-1, 3).astype(np.float32)
        return np.median(features, axis=0)

    def fit(self, frame, player_detections):
        features = []
        ids = []

        for player_id, detection in player_detections.items():
            feature = self._extract_features(frame, detection["box"])
            if feature is not None:
                features.append(feature)
                ids.append(player_id)

        if len(features) < self.n_teams:
            raise ValueError("Not enough player crops to assign teams")

        self.kmeans = KMeans(
            n_clusters=self.n_teams,
            random_state=self.random_state,
            n_init=20,
        )
        labels = self.kmeans.fit_predict(np.asarray(features))

        self.player_team = {
            player_id: int(label) + 1
            for player_id, label in zip(ids, labels)
        }
        self.team_colors = {
            team_id: self.kmeans.cluster_centers_[team_id - 1]
            for team_id in range(1, self.n_teams + 1)
        }

        return self.player_team

    def assign_team(self, frame, bbox):
        if self.kmeans is None:
            raise RuntimeError("TeamAssigner must be fitted before assignment")

        feature = self._extract_features(frame, bbox)
        if feature is None:
            return None

        return int(self.kmeans.predict([feature])[0]) + 1

    def assign_player_team(self, player_id, frame=None, bbox=None):
        if player_id in self.player_team:
            return self.player_team[player_id]

        if frame is None or bbox is None:
            return None

        team_id = self.assign_team(frame, bbox)
        if team_id is not None:
            self.player_team[player_id] = team_id
        return team_id

    def stabilize_assignments(self, history, min_votes=3):
        stable = {}
        for player_id, assignments in history.items():
            values = [value for value in assignments if value is not None]
            if len(values) < min_votes:
                continue
            stable[player_id] = Counter(values).most_common(1)[0][0]
        self.player_team.update(stable)
        return stable

    def update_from_tracks(self, frame, tracks):
        assignments = {}
        for player_id, detection in tracks.items():
            team_id = self.assign_player_team(
                player_id,
                frame,
                detection["box"],
            )
            assignments[player_id] = team_id
        return assignments

    def get_team_players(self, assignments):
        teams = defaultdict(list)
        for player_id, team_id in assignments.items():
            if team_id is not None:
                teams[team_id].append(player_id)
        return dict(teams)
