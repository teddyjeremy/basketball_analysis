import cv2


class BallDrawer:
    def __init__(self, color=(0, 165, 255), radius=8):
        self.color = color
        self.radius = radius

    def draw(self, frame, detection):
        output = frame.copy()
        if not detection:
            return output
        x1, y1, x2, y2 = detection["box"]
        center = (int((x1 + x2) / 2), int((y1 + y2) / 2))
        cv2.circle(output, center, self.radius, self.color, -1)
        return output
