import cv2


class PlayerDrawer:
    def __init__(self, team_colors=None, thickness=2):
        self.team_colors = team_colors or {1: (255, 80, 80), 2: (80, 80, 255)}
        self.thickness = thickness

    def draw(self, frame, tracks, team_assignments=None):
        output = frame.copy()
        team_assignments = team_assignments or {}

        for player_id, detection in tracks.items():
            x1, y1, x2, y2 = [int(value) for value in detection["box"]]
            team_id = team_assignments.get(player_id)
            color = self.team_colors.get(team_id, (255, 255, 255))
            cv2.rectangle(output, (x1, y1), (x2, y2), color, self.thickness)
            cv2.putText(
                output,
                f"P{player_id}",
                (x1, max(20, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA,
            )
        return output
