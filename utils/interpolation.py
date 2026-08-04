import numpy as np


def interpolate_missing(values, max_gap=10):
    result = list(values)
    valid = [index for index, value in enumerate(result) if value is not None]

    if len(valid) < 2:
        return result

    for left, right in zip(valid[:-1], valid[1:]):
        gap = right - left - 1
        if gap <= 0 or gap > max_gap:
            continue

        start = np.asarray(result[left], dtype=np.float32)
        end = np.asarray(result[right], dtype=np.float32)

        for offset in range(1, gap + 1):
            alpha = offset / (gap + 1)
            result[left + offset] = (
                start * (1.0 - alpha) + end * alpha
            ).tolist()

    return result
