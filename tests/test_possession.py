from ball_acquisition.ball_acquisition import BallAcquisition


def test_possession_assignment():
    tracker = BallAcquisition(max_distance=100)
    players = {
        1: {"box": [0, 0, 20, 40]},
        2: {"box": [300, 300, 320, 340]},
    }
    ball = {"box": [5, 25, 15, 35]}
    assert tracker.assign(players, ball) == 1
