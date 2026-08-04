import cv2


class TacticalViewDrawer:
    def __init__(self, team_colors=None, player_radius=7, ball_radius=5):
        self.team_colors = team_colors or {1: (255, 80, 80), 2: (80, 80, 255)}
        self.player_radius = player_radius
        self.ball_radius = ball_radius

    def draw_players(self, frame, positions, team_assignments=None):
        output = frame.copy()
        team_assignments = team_assignments or {}
        for player_id, position in positions.items():
            team_id = team_assignments.get(player_id)
            color = self.team_colors.get(team_id, (255, 255, 255))
            x, y = int(position[0]), int(position[1])
            cv2.circle(output, (x, y), self.player_radius, color, -1)
            cv2.putText(
                output,
                str(player_id),
                (x + 8, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                color,
                1,
                cv2.LINE_AA,
            )
        return output

    def draw_ball(self, frame, position):
        output = frame.copy()
        if position is not None:
            cv2.circle(
                output,
                (int(position[0]), int(position[1])),
                self.ball_radius,
                (0, 165, 255),
                -1,
            )
        return output
