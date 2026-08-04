class PassAndInterceptionDetector:
    def __init__(self, min_pass_frames=3, max_missing_frames=12):
        self.min_pass_frames = min_pass_frames
        self.max_missing_frames = max_missing_frames

    def detect(self, possession, team_assignments):
        events = []
        previous_player = None
        previous_team = None
        gap = 0

        for frame_index, player_id in enumerate(possession):
            if player_id is None:
                gap += 1
                if gap <= self.max_missing_frames:
                    continue
                previous_player = None
                previous_team = None
                continue

            team_id = team_assignments.get(player_id)
            if previous_player is not None and player_id != previous_player:
                if team_id == previous_team:
                    events.append(
                        {
                            "type": "pass",
                            "frame": frame_index,
                            "from_player": previous_player,
                            "to_player": player_id,
                            "team": team_id,
                        }
                    )
                elif team_id is not None and previous_team is not None:
                    events.append(
                        {
                            "type": "interception",
                            "frame": frame_index,
                            "from_player": previous_player,
                            "to_player": player_id,
                            "team": team_id,
                        }
                    )

            previous_player = player_id
            previous_team = team_id
            gap = 0

        return events

    def summarize(self, events):
        summary = {
            "passes": 0,
            "interceptions": 0,
            "events": len(events),
        }

        for event in events:
            if event["type"] == "pass":
                summary["passes"] += 1
            elif event["type"] == "interception":
                summary["interceptions"] += 1

        return summary
