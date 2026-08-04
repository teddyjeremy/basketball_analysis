import cv2


class CourtDrawer:
    def __init__(self, color=(255, 255, 255), thickness=2):
        self.color = color
        self.thickness = thickness

    def draw_keypoints(self, frame, points, confidence=None, threshold=0.5):
        output = frame.copy()
        for index, point in enumerate(points):
            if confidence is not None and confidence[index] < threshold:
                continue
            x, y = int(point[0]), int(point[1])
            cv2.circle(output, (x, y), 5, self.color, -1)
            cv2.putText(
                output,
                str(index),
                (x + 6, y - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                self.color,
                1,
                cv2.LINE_AA,
            )
        return output

    def draw_lines(self, frame, lines):
        output = frame.copy()
        for start, end in lines:
            cv2.line(
                output,
                tuple(map(int, start)),
                tuple(map(int, end)),
                self.color,
                self.thickness,
            )
        return output
