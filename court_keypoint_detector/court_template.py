import numpy as np


NBA_COURT_LENGTH_M = 28.6512
NBA_COURT_WIDTH_M = 15.24


def nba_court_keypoints():
    length = NBA_COURT_LENGTH_M
    width = NBA_COURT_WIDTH_M
    return np.asarray(
        [
            [0.0, 0.0],
            [length / 2.0, 0.0],
            [length, 0.0],
            [0.0, width / 2.0],
            [length, width / 2.0],
            [0.0, width],
            [length / 2.0, width],
            [length, width],
            [5.7912, width / 2.0],
            [length - 5.7912, width / 2.0],
            [5.7912, 1.6764],
            [5.7912, width - 1.6764],
            [length - 5.7912, 1.6764],
            [length - 5.7912, width - 1.6764],
        ],
        dtype=np.float32,
    )


def normalized_court_keypoints(length=NBA_COURT_LENGTH_M, width=NBA_COURT_WIDTH_M):
    points = nba_court_keypoints().copy()
    points[:, 0] /= length
    points[:, 1] /= width
    return points
