from itertools import permutations

import pytest
import numpy as np

from top2vec.elbow_finding import (
    find_elbow_index,
    get_distances_from_line,
    __euclidean_distance,
    __manhattan_distance,
    __uniform_distance,
)


# Make sure our metrics work the way we think
def test_euclidean_distance():
    assert __euclidean_distance(0, 0, 0, 0) == 0
    assert __euclidean_distance(2, 2, 2.5, 2.5) == (2 * ((2.5 - 2.0) ** 2)) ** 0.5
    assert __euclidean_distance(1, 1, 2.5, 2.5) == (2 * ((2.5 - 1.0) ** 2)) ** 0.5
    assert __euclidean_distance(0, 0, 2.5, 2.5) == (2 * (2.5**2)) ** 0.5
    assert __euclidean_distance(-2.5, -2.5, 2.5, 2.5) == (2 * (5**2)) ** 0.5
    assert __euclidean_distance(-2.5, 0, 2.5, 2.5) == ((5**2) + (2.5**2)) ** 0.5
    assert __euclidean_distance(0, -2.5, 2.5, 2.5) == ((5**2) + (2.5**2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, 2, 2) == (2 * ((2.5 - 2.0) ** 2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, 1, 1) == (2 * ((2.5 - 1.0) ** 2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, 0, 0) == (2 * (2.5**2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, -2.5, -2.5) == (2 * (5**2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, -2.5, 0) == ((5**2) + (2.5**2)) ** 0.5
    assert __euclidean_distance(2.5, 2.5, 0, -2.5) == ((5**2) + (2.5**2)) ** 0.5


def test_manhattan_distance():
    assert __manhattan_distance(0, 0, 0, 0) == 0
    assert __manhattan_distance(2, 2, 2.5, 2.5) == 1
    assert __manhattan_distance(1, 1, 2.5, 2.5) == 3
    assert __manhattan_distance(0, 0, 2.5, 2.5) == 5
    assert __manhattan_distance(-2.5, -2.5, 2.5, 2.5) == 10
    assert __manhattan_distance(-2.5, 0, 2.5, 2.5) == 7.5
    assert __manhattan_distance(0, -2.5, 2.5, 2.5) == 7.5
    assert __manhattan_distance(2.5, 2.5, 2, 2) == 1
    assert __manhattan_distance(2.5, 2.5, 1, 1) == 3
    assert __manhattan_distance(2.5, 2.5, 0, 0) == 5
    assert __manhattan_distance(2.5, 2.5, -2.5, -2.5) == 10
    assert __manhattan_distance(2.5, 2.5, -2.5, 0) == 7.5
    assert __manhattan_distance(2.5, 2.5, 0, -2.5) == 7.5


def test_uniform_distance():
    assert __uniform_distance(0, 0, 0, 0) == 0
    assert __uniform_distance(2, 2, 2.5, 2.5) == 0.5
    assert __uniform_distance(1, 1, 2.5, 2.5) == 1.5
    assert __uniform_distance(0, 0, 2.5, 2.5) == 2.5
    assert __uniform_distance(-2.5, -2.5, 2.5, 2.5) == 5
    assert __uniform_distance(-2.5, 0, 2.5, 2.5) == 5
    assert __uniform_distance(0, -2.5, 2.5, 2.5) == 5
    assert __uniform_distance(2.5, 2.5, 2, 2) == 0.5
    assert __uniform_distance(2.5, 2.5, 1, 1) == 1.5
    assert __uniform_distance(2.5, 2.5, 0, 0) == 2.5
    assert __uniform_distance(2.5, 2.5, -2.5, -2.5) == 5
    assert __uniform_distance(2.5, 2.5, -2.5, 0) == 5
    assert __uniform_distance(2.5, 2.5, 0, -2.5) == 5


def compare_numpy_arrays(array_a, array_b):
    if array_a is None and array_b is None:
        return True
    elif array_a is None or array_b is None:
        return False

    if array_a.size == 0 and array_b.size == 0:
        return True
    elif array_a.size == 0 or array_b.size == 0:
        return False

    if array_a.shape != array_b.shape:
        return False
    return (array_a == array_b).all()


def distances_from_line_helper(metric, expected_distance):
    # If we give it an empty list then things should
    # be an empty array
    assert compare_numpy_arrays(get_distances_from_line([], 1, 0), np.array([]))

    y_int = 5
    slope = -1

    # Items which follow the slope exactly should be all 0 for all metrics
    assert compare_numpy_arrays(
        get_distances_from_line([5, 4, 3, 2, 1], slope, y_int, metric), np.zeros(5)
    )
    # Order DOES matter
    assert not compare_numpy_arrays(
        get_distances_from_line([3, 4, 5, 2, 1], slope, y_int, metric), np.zeros(5)
    )
    assert compare_numpy_arrays(
        get_distances_from_line([5, 4, 2, 2, 1], slope, y_int, metric),
        np.array([0, 0, expected_distance, 0, 0]),
    )
    # Should be symmetrical
    # This flips around the line, so we need to say we want everything
    assert compare_numpy_arrays(
        get_distances_from_line(
            [5, 4, 2, 3, 1], slope, y_int, metric, first_elbow=False
        ),
        np.array([0, 0, expected_distance, expected_distance, 0]),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(
            [5, 4, 2, 3, 1], slope, y_int, metric, first_elbow=True
        ),
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(
            [8, 7, 7, 5, 4, 2, 3, 1], slope, 8, metric, first_elbow=True
        ),
        np.array([0, 0, expected_distance, 0, 0, 0, 0, 0]),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(
            [8, 7, 7, 5, 4, 4, 1, 1], slope, 8, metric, first_elbow=True
        ),
        np.array([0, 0, expected_distance, 0, 0, expected_distance, 0, 0]),
    )

    # What if we did a numpy array rather than just a list of numbers?
    assert compare_numpy_arrays(
        get_distances_from_line(np.array([5, 4, 3, 2, 1]), slope, y_int, metric),
        np.zeros(5),
    )
    assert not compare_numpy_arrays(
        get_distances_from_line(np.array([3, 4, 5, 2, 1]), slope, y_int, metric),
        np.zeros(5),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(np.array([5, 4, 2, 2, 1]), slope, y_int, metric),
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(
            np.array([5, 4, 2, 3, 1]), slope, y_int, metric, first_elbow=False
        ),
        np.array([0, 0, expected_distance, expected_distance, 0]),
    )

    # But now what about when we are given something that flips across the line?
    assert compare_numpy_arrays(
        get_distances_from_line(
            np.array([5, 4, 2, 3, 1]), slope, y_int, metric, first_elbow=True
        ),
        np.array([0, 0, expected_distance, 0, 0]),
    )

    # What about the slope of zero? Should be the same for all metrics.
    assert compare_numpy_arrays(
        get_distances_from_line([1, 2, 3], 0, 1, metric), np.array([0, 1, 2])
    )
    assert compare_numpy_arrays(
        get_distances_from_line(np.array([5, 4, 2, 3, 1]), 0, 0, metric),
        np.array([5, 4, 2, 3, 1]),
    )


def test_get_distances_from_line_euclidean():
    # this distance should be between (2,2) and (2.5, 2.5)
    distances_from_line_helper("euclidean", (2 * ((2.5 - 2.0) ** 2)) ** 0.5)


def test_get_distances_from_line_manhattan():
    # this distance should be between (2,2) and (2.5, 2.5)
    distances_from_line_helper("manhattan", 1)


def test_get_distances_from_line_uniform():
    # this distance should be between (2,2) and (2.5, 2.5)
    distances_from_line_helper("uniform", 0.5)


def test_find_elbow_index():
    # Tell me what the farthest shortest distance from the line is!
    assert find_elbow_index([]) == -1
    assert find_elbow_index(None) == -1
    assert find_elbow_index([0, 0, 0, 0]) == -1
    assert find_elbow_index([0]) == -1
    assert find_elbow_index([0, 0]) == -1
    assert find_elbow_index([1]) == 0
    assert find_elbow_index([1, 111111111]) == 0
    with pytest.raises(ValueError):
        find_elbow_index([[1, 2, 22]])

    def elbow_index_helper(base_data, expected):
        # Make sure that the order doesn't matter here and neither
        # does NP array or list
        perms = permutations(base_data)
        # In 2d spaces the end result of the different norms should be identical
        for metric in ["euclidean", "manhattan", "uniform"]:
            for instance in perms:
                assert find_elbow_index(instance, metric) == expected
                assert find_elbow_index(np.array(instance), metric) == expected

    # Anything of size 1 or 2 will return the first
    elbow_index_helper([2], 0)
    elbow_index_helper([0.022], 0)
    elbow_index_helper([2, 8908980], 0)
    elbow_index_helper([0.022, 123123.1233], 0)

    # If everything is equal we should get the first value
    elbow_index_helper([5, 4, 3, 2, 1], 0)

    # Now a real value
    elbow_index_helper([5, 4, 2, 2, 1], 2)

    # What if we have equidistent points?
    elbow_index_helper([5, 4, 2, 1, 1], 2)

    # What if we have equidistent points?
    elbow_index_helper([5, 4, 2, 1, 1, 0], 2)

    elbow_index_helper([7, 7, 5, 4, 4, 4, 1, 1], 1)

    # Now a very bad graph with many elbows
    flipper = [
        8,
        7,
        7,
        7,
        7,
        7,
        5,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        -1,
        -1,
    ]
    assert find_elbow_index(flipper, first_elbow=True) == 1
    assert find_elbow_index(flipper, first_elbow=False) == 7

    flipper2 = flipper
    flipper2[0] = 7
    assert find_elbow_index(flipper2, first_elbow=True) == 5
    assert find_elbow_index(flipper2, first_elbow=False) == 7


def test_bad_values():
    # Strict elbow finding on this returns sub-optimal values due to the massive drop between the first and second item
    # Make this work better!
    values = np.array(
        [
            0.7015004,
            0.21765897,
            0.21430951,
            0.1833846,
            0.17638168,
            0.17076348,
            0.15859194,
            0.1492473,
            0.14306031,
            0.14136967,
            0.13891019,
            0.13815553,
            0.13220927,
            0.13091132,
            0.12623666,
            0.11688363,
            0.11273225,
            0.10439977,
            0.10002852,
            0.0990672,
            0.09879605,
            0.09669335,
            0.09599731,
            0.08999116,
            0.08903309,
            0.0889603,
            0.08854498,
            0.08833782,
            0.07953031,
            0.07422821,
            0.0740669,
            0.07364371,
            0.07136905,
            0.07084752,
            0.06936937,
            0.06931616,
            0.06761124,
            0.06568824,
            0.06138799,
            0.06041424,
            0.0602054,
            0.05955762,
            0.05785264,
            0.05535707,
            0.05316744,
            0.05253472,
            0.05048357,
            0.04145124,
            0.03034239,
            0.02655142,
        ]
    )
    assert find_elbow_index(values) == 0
