from itertools import permutations
from random import shuffle

import pytest
import numpy as np

from top2vec.cutoff_heuristics.cutoff_heuristics import (
    RECURSIVE_ELBOW_HEURISTIC_STR,
    find_elbow_index,
    find_cutoff,
    get_distances_from_line,
    _get_shifted_second_derivative,
    ELBOW_HEURISTIC_STR,
    DERIVATIVE_HEURISTIC_STR,
    AVERAGE_HEURISTIC_STR,
    SUPPORTED_HEURISTICS,
)
from top2vec.types import LineDistances


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


def test_compare_numpy_arrays():
    assert compare_numpy_arrays(None, None)
    assert not compare_numpy_arrays(np.array([1, 2, 3]), None)
    assert compare_numpy_arrays(np.array([]), np.array([]))
    assert compare_numpy_arrays(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not compare_numpy_arrays(np.array([1, 2, 3]), np.array([]))
    assert not compare_numpy_arrays(np.array([[1, 2, 3]]), np.array([1, 2, 3]))
    assert not compare_numpy_arrays(np.array([[1, 1], [1, 1]]), np.array([1, 1, 1, 1]))


def test_distances_from_line():
    # If we give it an empty list then things should
    # be an empty array
    res_tup = get_distances_from_line([], 1, 0)
    assert compare_numpy_arrays(res_tup.distances, np.array([]))
    assert res_tup.truncation_index == -1
    assert not res_tup.is_truncated

    y_int = 5
    slope = -1
    expected_distance = 1

    # Items which follow the slope exactly should be all 0 for all metrics
    res_tup = get_distances_from_line([5, 4, 3, 2, 1], slope, y_int)
    assert compare_numpy_arrays(res_tup.distances, np.zeros(5))
    # Exactly same as line should be true
    assert res_tup.first_elbow_above_line
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 4
    # Don't really care about values, care about signs
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0, np.array([False, False, False, False, False])
    )

    # Order DOES matter
    res_tup = get_distances_from_line([3, 4, 5, 2, 1], slope, y_int)
    assert not compare_numpy_arrays(res_tup.distances, np.zeros(5))
    assert not res_tup.first_elbow_above_line
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 1
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0, np.array([True, False, False, False, False])
    )

    res_tup = get_distances_from_line([5, 4, 2, 2, 1], slope, y_int)
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert not res_tup.first_elbow_above_line
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 4
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0, np.array([False, False, True, False, False])
    )

    # Should be symmetrical
    # This flips around the line, so we need to say we want everything
    res_tup = get_distances_from_line([5, 4, 2, 3, 1], slope, y_int, first_elbow=False)
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, expected_distance, 0]),
    )
    assert not res_tup.first_elbow_above_line
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 4
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0, np.array([False, False, True, False, False])
    )

    res_tup = get_distances_from_line([5, 4, 2, 3, 1], slope, y_int, first_elbow=True)
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert not res_tup.first_elbow_above_line
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 2
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0, np.array([False, False, True, False, False])
    )

    res_tup = get_distances_from_line(
        [8, 7, 7, 5, 4, 2, 3, 1], slope, 8, first_elbow=True
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, 0, 0, 0, 0, 0]),
    )
    assert res_tup.first_elbow_above_line
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 4
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0,
        np.array([False, False, False, False, False, False, False, False]),
    )
    assert compare_numpy_arrays(
        res_tup.y_deltas > 0,
        np.array([False, False, True, False, False, False, False, False]),
    )

    res_tup = get_distances_from_line(
        [8, 7, 7, 5, 4, 2, 3, 1], slope, 8, first_elbow=False
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array(
            [0, 0, expected_distance, 0, 0, expected_distance, expected_distance, 0]
        ),
    )
    assert res_tup.first_elbow_above_line
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 7
    assert compare_numpy_arrays(
        res_tup.y_deltas < 0,
        np.array([False, False, False, False, False, True, False, False]),
    )
    assert compare_numpy_arrays(
        res_tup.y_deltas > 0,
        np.array([False, False, True, False, False, False, True, False]),
    )

    res_tup = get_distances_from_line(
        [8, 7, 7, 5, 4, 4, 1, 1], slope, 8, first_elbow=True
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, 0, 0, expected_distance, 0, 0]),
    )
    assert res_tup.first_elbow_above_line
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 5

    # What if we did a numpy array rather than just a list of numbers?
    res_tup = get_distances_from_line(np.array([5, 4, 3, 2, 1]), slope, y_int)
    assert compare_numpy_arrays(
        res_tup.distances,
        np.zeros(5),
    )
    assert not compare_numpy_arrays(
        get_distances_from_line(np.array([3, 4, 5, 2, 1]), slope, y_int).distances,
        np.zeros(5),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(np.array([5, 4, 2, 2, 1]), slope, y_int).distances,
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert compare_numpy_arrays(
        get_distances_from_line(
            np.array([5, 4, 2, 3, 1]), slope, y_int, first_elbow=False
        ).distances,
        np.array([0, 0, expected_distance, expected_distance, 0]),
    )

    # But now what about when we are given something that flips across the line?
    res_tup = get_distances_from_line(
        np.array([5, 4, 2, 3, 1]), slope, y_int, first_elbow=True
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([0, 0, expected_distance, 0, 0]),
    )
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 2
    assert not res_tup.first_elbow_above_line

    # What about the slope of zero? Should be the same for all metrics.
    res_tup = get_distances_from_line([1, 2, 3], 0, 1)
    assert compare_numpy_arrays(res_tup.distances, np.array([0, 1, 2]))
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 2
    assert res_tup.first_elbow_above_line

    res_tup = get_distances_from_line(np.array([5, 4, 2, 3, 1]), 0, 0)
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([5, 4, 2, 3, 1]),
    )
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 4
    assert res_tup.first_elbow_above_line

    res_tup = get_distances_from_line(
        np.array([5, 4, 0, -1, 0]), 0, 0, first_elbow=True
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([5, 4, 0, 0, 0]),
    )
    assert res_tup.is_truncated
    assert res_tup.truncation_index == 2
    assert res_tup.first_elbow_above_line

    res_tup = get_distances_from_line(
        np.array([-5, -4, 0, 1, 0]), 0, 0, first_elbow=False
    )
    assert compare_numpy_arrays(
        res_tup.distances,
        np.array([5, 4, 0, 1, 0]),
    )
    assert not res_tup.is_truncated
    assert res_tup.truncation_index == 4
    assert not res_tup.first_elbow_above_line


def test_find_cutoff():
    with pytest.raises(ValueError):
        find_cutoff([], cutoff_heuristic="Fake Data")
    with pytest.raises(ValueError):
        find_cutoff([], cutoff_heuristic=None)

    for heuristic in [
        ELBOW_HEURISTIC_STR,
        DERIVATIVE_HEURISTIC_STR,
        AVERAGE_HEURISTIC_STR,
    ]:
        assert find_cutoff([], cutoff_heuristic=heuristic) == -1
        assert find_cutoff(None, cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0, 0, 0, 0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0, 0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([1], cutoff_heuristic=heuristic) == 0
        assert find_cutoff([1, 111111111], cutoff_heuristic=heuristic) == 0
        assert (
            find_cutoff(
                [10, 2, 1, 0], cutoff_heuristic=heuristic, max_first_delta=0.001
            )
            == 0
        )
        test_data = [20, 19, 18, 17, 16, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1]
        assert (
            find_cutoff(
                test_data,
                cutoff_heuristic=heuristic,
                first_elbow=True,
                below_line_exclusive=False,
            )
            == 5
        )
        assert (
            find_cutoff(
                test_data,
                cutoff_heuristic=heuristic,
                first_elbow=True,
                below_line_exclusive=True,
            )
            == 4
        )
        shuffle(test_data)
        assert (
            find_cutoff(
                test_data,
                cutoff_heuristic=heuristic,
                first_elbow=True,
                below_line_exclusive=False,
            )
            == 5
        )
        assert (
            find_cutoff(
                test_data,
                cutoff_heuristic=heuristic,
                first_elbow=True,
                below_line_exclusive=True,
            )
            == 4
        )


def test_recursive_elbow_index():
    # Anything which results in index 1 or less should be the same
    for heuristic in [ELBOW_HEURISTIC_STR, RECURSIVE_ELBOW_HEURISTIC_STR]:
        assert find_cutoff([], cutoff_heuristic=heuristic) == -1
        assert find_cutoff(None, cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0, 0, 0, 0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([0, 0], cutoff_heuristic=heuristic) == -1
        assert find_cutoff([1], cutoff_heuristic=heuristic) == 0
        assert find_cutoff([1, 111111111], cutoff_heuristic=heuristic) == 0
        assert (
            find_cutoff(
                [10, 2, 1, 0], cutoff_heuristic=heuristic, max_first_delta=0.001
            )
            == 0
        )
    test_data = [20, 19, 17, 17, 16, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1]
    first_pass = find_cutoff(
        test_data,
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=False,
    )
    assert first_pass == 5
    second_pass = find_cutoff(
        test_data[: first_pass + 1],
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=False,
    )
    assert second_pass == find_cutoff(
        test_data,
        cutoff_heuristic=RECURSIVE_ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=False,
    )
    assert second_pass == 4

    first_pass = find_cutoff(
        test_data,
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=True,
    )
    assert first_pass == 4
    second_pass = find_cutoff(
        test_data[: first_pass + 1],
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=True,
    )
    assert second_pass == find_cutoff(
        test_data,
        cutoff_heuristic=RECURSIVE_ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=True,
    )
    assert second_pass == 1

    test_data = [20, 5, 0]
    first_pass = find_cutoff(
        test_data,
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=False,
        max_first_delta=None,
    )
    assert first_pass == 1
    assert 1 == find_cutoff(
        test_data,
        cutoff_heuristic=RECURSIVE_ELBOW_HEURISTIC_STR,
        first_elbow=True,
        below_line_exclusive=False,
        max_first_delta=None,
    )


def test_derivative_index():
    base_data = [20, 19, 18, 17, 16, 10, 9, 8, 7, 6, 6, 6, 6, 6, 6, 5, 4, 3, 2, 1]
    assert (
        find_cutoff(base_data, cutoff_heuristic=ELBOW_HEURISTIC_STR)
    ) == find_cutoff(base_data, cutoff_heuristic=DERIVATIVE_HEURISTIC_STR)
    assert (
        find_cutoff(base_data, cutoff_heuristic=AVERAGE_HEURISTIC_STR)
    ) == find_cutoff(base_data, cutoff_heuristic=DERIVATIVE_HEURISTIC_STR)

    derivative_data = [
        20,
        19,
        18,
        17,
        16,
        14,
        13,
        9,
        7,
        6,
        6,
        6,
        6,
        6,
        6,
        5,
        4,
        3,
        2,
        1,
    ]
    assert (
        find_cutoff(derivative_data, cutoff_heuristic=ELBOW_HEURISTIC_STR) - 1
    ) == find_cutoff(derivative_data, cutoff_heuristic=DERIVATIVE_HEURISTIC_STR)
    assert (
        find_cutoff(derivative_data, cutoff_heuristic=AVERAGE_HEURISTIC_STR)
    ) == find_cutoff(derivative_data, cutoff_heuristic=DERIVATIVE_HEURISTIC_STR)


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
        for instance in perms:
            assert find_elbow_index(instance, below_line_exclusive=False) == expected
            assert (
                find_elbow_index(np.array(instance), below_line_exclusive=False)
                == expected
            )

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
    assert find_elbow_index(flipper, first_elbow=True, below_line_exclusive=False) == 1
    # Immediately flips underneath
    assert find_elbow_index(flipper, first_elbow=True, below_line_exclusive=True) == 0
    assert find_elbow_index(flipper, first_elbow=False, below_line_exclusive=False) == 7
    assert find_elbow_index(flipper, first_elbow=False, below_line_exclusive=True) == 6

    flipper2 = flipper
    flipper2[0] = 7
    assert find_elbow_index(flipper2, first_elbow=True, below_line_exclusive=False) == 5
    assert find_elbow_index(flipper2, first_elbow=True, below_line_exclusive=True) == 5
    assert (
        find_elbow_index(flipper2, first_elbow=False, below_line_exclusive=False) == 7
    )
    # NOTE: This is an interesting test case
    # The first elbow is above the line, but the actual point being returned is below the line
    assert find_elbow_index(flipper2, first_elbow=False, below_line_exclusive=True) == 6


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

    # This one is testing that we are exclusive when below the curve
    values = np.array(
        [
            0.87387407,
            0.8490747,
            0.83483994,
            0.80989516,
            0.45845926,
            0.45052826,
            0.44408453,
            0.4278804,
            0.4249642,
            0.41800153,
            0.415339,
            0.40166456,
            0.4011852,
            0.3939832,
            0.38374978,
            0.3823452,
            0.37897837,
            0.37643087,
            0.37551993,
            0.37453377,
            0.37433827,
            0.3703115,
            0.36441594,
            0.3591773,
            0.35516483,
            0.35450447,
            0.35071152,
            0.34965813,
            0.3412869,
            0.3399775,
            0.33868152,
            0.33400196,
            0.315881,
        ]
    )
    assert find_elbow_index(values) == 3


def test_slid_second_derivative():
    test = np.array([0, 1, 2, 3, 4, 5, 6])
    # first_der = np.array([0, 1, 1, 1, 1, 1, 1])
    second_der = np.array([1, 0, 0, 0, 0, 0, 0])
    with pytest.raises(ValueError):
        assert compare_numpy_arrays(
            _get_shifted_second_derivative(test, False, 20), second_der
        )
    assert compare_numpy_arrays(
        _get_shifted_second_derivative(test, False, test.size - 1), second_der
    )

    test = np.array([0, 1, 2, 6, 7, 1, 0])
    # first_der = np.array([0, 1, 1, 4, 1, -6, -1])
    second_der = np.array([1, 0, 3, 3, 7, 5, 0])
    with pytest.raises(ValueError):
        assert compare_numpy_arrays(
            _get_shifted_second_derivative(test, False, 20), second_der
        )
    assert compare_numpy_arrays(
        _get_shifted_second_derivative(test, False, test.size - 1), second_der
    )
    # Should get raw values one beyond truncation index
    assert compare_numpy_arrays(
        _get_shifted_second_derivative(test, True, 5), second_der[:6]
    )
    assert compare_numpy_arrays(
        _get_shifted_second_derivative(test, True, 4), second_der[:5]
    )
    with pytest.raises(ValueError):
        assert compare_numpy_arrays(
            _get_shifted_second_derivative(test, True, 20), second_der
        )
    # This isn't using y_deltas at all
    test_distances = LineDistances(test, np.array([1, 1, 1, 1, 1, 1, 1]), True, 4, True)
    assert compare_numpy_arrays(
        _get_shifted_second_derivative(
            test_distances.distances,
            test_distances.is_truncated,
            test_distances.truncation_index,
        ),
        second_der[:5],
    )


# Found a case where doing sorted versus unsorted gave a different result
@pytest.mark.parametrize("heuristic", SUPPORTED_HEURISTICS)
def test_sorted(heuristic):
    TEST_DATA = np.array(
        [
            0.4477413296699524,
            0.5869248509407043,
            0.6227058172225952,
            0.08470261096954346,
            0.24712997674942017,
            0.6084471344947815,
            0.405947208404541,
            0.4543569087982178,
            0.3563116788864136,
            0.21436893939971924,
            0.5348855257034302,
            0.5431680679321289,
            0.11027193069458008,
            -0.0008113384246826172,
            0.03843200206756592,
            0.7738500237464905,
            0.4038645625114441,
            0.7747093439102173,
            0.5071402788162231,
            0.42496275901794434,
            0.3468637466430664,
            0.8517152070999146,
            0.15013813972473145,
            0.25401294231414795,
            0.14735198020935059,
            0.7237462997436523,
            0.27071553468704224,
            0.3637889623641968,
            0.050290584564208984,
            0.23538267612457275,
            0.14029526710510254,
            0.3706018924713135,
            0.10026603937149048,
            0.13935357332229614,
            0.06049245595932007,
            -0.041458845138549805,
            0.007721960544586182,
            0.418732225894928,
            0.2730292081832886,
            0.5806816220283508,
            0.41289085149765015,
            0.6815071105957031,
            0.37891077995300293,
            0.6017580032348633,
            0.6396514177322388,
            0.38906025886535645,
            0.26992255449295044,
            0.5818625688552856,
            0.5946063995361328,
            0.48077285289764404,
            0.6625726222991943,
            0.17979896068572998,
            0.3204556703567505,
            0.6141220331192017,
            0.5092817544937134,
            0.43334317207336426,
            0.21307039260864258,
            0.49678975343704224,
            0.28558874130249023,
            0.38742560148239136,
            0.21473819017410278,
            0.46951723098754883,
            0.4187730550765991,
            0.5146691799163818,
            0.2743796110153198,
            0.393377423286438,
            0.7049881815910339,
            0.3618965148925781,
            0.34342050552368164,
            0.18461889028549194,
            0.048503220081329346,
            0.340198278427124,
            0.25806427001953125,
            0.4905596971511841,
            0.14241480827331543,
            0.34514325857162476,
            0.5485992431640625,
            0.27928632497787476,
            0.3636154532432556,
            0.659136950969696,
            0.2568431496620178,
            0.31281358003616333,
            0.6339702010154724,
            0.21866953372955322,
            0.4574323296546936,
            0.40434110164642334,
            0.4843582510948181,
            0.36593174934387207,
            0.4263142943382263,
            0.5326312780380249,
            0.37244921922683716,
            0.4573936462402344,
            0.4374304413795471,
            0.45709770917892456,
            0.42226552963256836,
            0.1188044548034668,
            0.2705400586128235,
            0.23354381322860718,
            0.3780297040939331,
            0.21478241682052612,
            0.27042895555496216,
        ]
    )
    assert find_cutoff(TEST_DATA, cutoff_heuristic=heuristic) == find_cutoff(
        np.flip(np.sort(TEST_DATA)), cutoff_heuristic=heuristic
    )
