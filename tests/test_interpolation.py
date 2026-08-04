from utils.interpolation import interpolate_missing


def test_interpolate_missing_values():
    result = interpolate_missing([(0, 0), None, (2, 2)])
    assert result[1] == [1.0, 1.0]


def test_long_gap_is_preserved():
    values = [(0, 0)] + [None] * 11 + [(12, 12)]
    result = interpolate_missing(values, max_gap=10)
    assert all(value is None for value in result[1:-1])
