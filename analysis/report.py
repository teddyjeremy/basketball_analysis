from collections import Counter

import numpy as np


def _json_safe(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def build_report(possession, events, player_metrics, team_assignments):
    team_possession = Counter()
    player_possession = Counter(
        value for value in possession if value is not None
    )

    for player_id, count in player_possession.items():
        team_id = team_assignments.get(player_id)
        if team_id is not None:
            team_possession[team_id] += count

    total = sum(team_possession.values())
    possession_percentage = (
        {
            str(team): count / total * 100.0
            for team, count in team_possession.items()
        }
        if total
        else {}
    )

    passes = [event for event in events if event["type"] == "pass"]
    interceptions = [
        event for event in events if event["type"] == "interception"
    ]

    report = {
        "team_possession_percentage": possession_percentage,
        "player_possession_frames": dict(player_possession),
        "passes": passes,
        "interceptions": interceptions,
        "total_passes": len(passes),
        "total_interceptions": len(interceptions),
        "player_metrics": player_metrics,
    }
    return _json_safe(report)
