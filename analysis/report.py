from collections import Counter


def build_report(possession, events, player_metrics, team_assignments):
    team_possession = Counter()
    player_possession = Counter(value for value in possession if value is not None)

    for player_id, count in player_possession.items():
        team_id = team_assignments.get(player_id)
        if team_id is not None:
            team_possession[team_id] += count

    total = sum(team_possession.values())
    possession_percentage = {
        team: count / total * 100.0
        for team, count in team_possession.items()
    } if total else {}

    passes = [event for event in events if event["type"] == "pass"]
    interceptions = [event for event in events if event["type"] == "interception"]

    return {
        "team_possession_percentage": possession_percentage,
        "passes": passes,
        "interceptions": interceptions,
        "total_passes": len(passes),
        "total_interceptions": len(interceptions),
        "player_metrics": player_metrics,
    }
