import cv2


class PossessionDrawer:
    def __init__(self, color=(0, 255, 255), radius=10):
        self.color = color
        self.radius = radius

    def draw(self, frame, tracks, possession_player):
        output = frame.copy()
        if possession_player is None or possession_player not in tracks:
            return output

        x1, y1, x2, y2 = [int(value) for value in tracks[possession_player]["box"]]
        center = (int((x1 + x2) / 2), y2)
        cv2.circle(output, center, self.radius, self.color, 2)
        return output
