"""Allow finding an elbow for cut-off heuristics.

Will eventually be rolled directly into Top2Vec.

Author: Shawn
License: BSD 3 clause

Notes
-----
Heuristics - assuming we are operating on `values`

An index of -1 means "nothing for cutoff".
Otherwise incices are assumed to be inclusive.

All heuristics go through the following algorithm first:
* If `values` is empty or all-zero: return -1
* If `values.size <= 2`: return 0
* If `max_first_delta` provided and the value change from index 0 to 1
  as a proportion of the total value change from index 0 to -1 is greater than
  or equal to `max_first_delta`: return 0

From here the individual heuristics are run
* elbow finding

* shifted_derivative
"""
from typing import Optional, NamedTuple
import numpy as np
from numpy.typing import NDArray, ArrayLike

# Prevent typos
__EUCLIDEAN_STR = "euclidean"
__MANHATTAN_STR = "manhattan"
__UNIFORM_STR = "uniform"
__RAW_Y_STR = "raw-y"

ELBOW_HEURISTIC_STR = "elbow"
DERIVATIVE_HEURISTIC_STR = "shifted_derivative"
AVERAGE_HEURISTIC_STR = "average"
SUPPORTED_HEURISTICS = [
    ELBOW_HEURISTIC_STR,
    DERIVATIVE_HEURISTIC_STR,
    AVERAGE_HEURISTIC_STR,
]


def __euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L2 distance between two points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def __manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L1 distance between two points"""
    return abs(x1 - x2) + abs(y1 - y2)


def __uniform_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L-infinity distance between two points"""
    return max(abs(x1 - x2), abs(y1 - y2))


def __raw_y_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the y difference (non-absolute)"""
    return y2 - y1


class LineDistances(NamedTuple):
    """Represents multiple data points about distance from a line.

    Attributes
    ----------
    distances: NdArray[np.float64]
        The shortest distance between value[index i] and any point
        along the provided line.
    is_truncated: bool
        If true: `first_elbow` was True and the values curve crossed
        the provided line at least once.
    truncation_index: int
        Last index before values curve first crosses line.
        All values after this in `distances` will be 0.
    first_elbow_above_line: bool
        Was the value curve above (or equal to) the provided line?
        Generally can be thought of as inclusive for
        whether to keep an elbow point.
    """

    distances: NDArray[np.float64]
    y_deltas: NDArray[np.float64]
    is_truncated: bool
    truncation_index: int
    first_elbow_above_line: bool


def get_distances_from_line(
    values: ArrayLike,
    comparison_slope: float,
    comparison_y_intercept: float,
    metric: str = __MANHATTAN_STR,
    first_elbow: bool = True,
) -> LineDistances:
    """Finds the shortest distance for all provided values from a provided line.

    Handles if the shortest distance to the line actually happens between
    two regularly listed points with integer X values.

    Parameters
    ----------
    values : ArrayLike
        A 1d array (or list) of y values such that index 0 is y(0).
    comparison_slope : float
        The slope of the line to compare values with.
    comparison_y_intercept : float
        The y intercept of the line to compare values with.
    metric: str (Optional default "manhattan")
        Which distance metric to use when comparing with the line.
        One of ("euclidean", "manhattan", "uniform").
    first_elbow: bool (default True)
        If true: computation will stop if y values change sign
        relative to the provided line.
        Only values prior to the sign flip will be returned.
        Intended to ensure that the elbow discovered is the first
        within a graph.

    Returns
    -------
    LineDistances
        The distances as well as additional metadata.
    """
    if metric == __EUCLIDEAN_STR:
        dist_fun = __euclidean_distance
    elif metric == __MANHATTAN_STR:
        dist_fun = __manhattan_distance
    elif metric == __UNIFORM_STR:
        dist_fun = __uniform_distance
    elif metric == __RAW_Y_STR:
        dist_fun = __raw_y_distance
    else:
        raise ValueError(
            f"Illegal metric - '{metric}'.\
           Must be one of [{__EUCLIDEAN_STR}, {__MANHATTAN_STR}, {__UNIFORM_STR}]"
        )
    try:
        n_elements = values.size
    except AttributeError:
        # handed a list rather than a numpy array
        n_elements = len(values)
    distances = np.zeros(n_elements)
    y_deltas = np.zeros(n_elements)
    to_examine = range(n_elements)

    perp_slope = comparison_slope * -1
    # only compute this once
    divisor = comparison_slope - perp_slope
    # TODO: look at np.vectorize
    # NOTE: Haven't been able to find a case yet where just comparing against the
    # raw y value doesn't give us the answer we want.
    was_positive_y = None
    truncation_index = n_elements - 1
    for x in to_examine:
        instance_y = values[x]
        # special case: slope of 0
        if divisor == 0:
            comparison_x = x
            comparison_y = comparison_y_intercept
        else:
            # Which y value are we computing distance for
            instance_y_intercept = instance_y - perp_slope * x

            comparison_x = (instance_y_intercept - comparison_y_intercept) / divisor
            comparison_y = comparison_slope * comparison_x + comparison_y_intercept

        # Rather than computing the true y-delta we are
        # going to re-use the closest point on the line
        # The sign of the delta should still be the same.
        y_dist = instance_y - comparison_y
        if y_dist != 0:
            if was_positive_y is None:
                was_positive_y = y_dist > 0

            if first_elbow:
                # NOTE: Depending on how this is parallelized (if at all)
                # it may make sense to have the bail-out in the calling
                # function
                if was_positive_y is None:
                    was_positive_y = y_dist > 0
                elif (was_positive_y and y_dist < 0) or (
                    not was_positive_y and y_dist > 0
                ):
                    truncation_index = x - 1
                    break

        distances[x] = dist_fun(x, instance_y, comparison_x, comparison_y)
        y_deltas[x] = y_dist
    return LineDistances(
        distances,
        y_deltas,
        truncation_index != n_elements - 1,
        truncation_index,
        was_positive_y is None or was_positive_y,
    )


def __edge_cases(sorted_values: ArrayLike, max_first_delta: Optional[float] = 0.33):
    """Handle edge cases prior to finding cut-off index"""
    if len(sorted_values.shape) != 1:
        raise ValueError("Values must be a 1-D Array")
    if sorted_values.size == 0 or np.count_nonzero(sorted_values) == 0:
        return -1
    elif sorted_values.size <= 2:
        return 0
    rise = sorted_values[-1] - sorted_values[0]
    if max_first_delta is not None:
        # determine percent of total drop contained between bin 0 and 1
        percent_delta = (sorted_values[1] - sorted_values[0]) / rise
        if percent_delta >= max_first_delta:
            return 0
    return None


def find_cutoff(
    values: ArrayLike,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    distance_metric: str = __MANHATTAN_STR,
    first_elbow: bool = True,
    max_first_delta: Optional[float] = 0.33,
    below_line_exclusive: bool = True,
):
    """Finds the cutoff index (inclusive) in a series of real values.

    Parameters
    ----------
    below_line_exclusive: bool (Optional default True)
        If true then result indices which are from an elbow below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    """

    if cutoff_heuristic not in SUPPORTED_HEURISTICS:
        raise ValueError(
            f"Unsupported heuristic selected: {cutoff_heuristic}. Must be one of {SUPPORTED_HEURISTICS}."
        )
    if values is None:
        return -1

    # Make sure the provided values are a sorted numpy array
    sorted_values = -np.sort(-np.array(values))
    edge_case_index = __edge_cases(
        sorted_values=sorted_values, max_first_delta=max_first_delta
    )
    if edge_case_index is not None:
        return edge_case_index

    rise = sorted_values[-1] - sorted_values[0]

    slope = rise / (sorted_values.size - 1)
    y_intercept = sorted_values[0]
    distances_tuple = get_distances_from_line(
        sorted_values,
        slope,
        y_intercept,
        metric=distance_metric,
        first_elbow=first_elbow,
    )

    if cutoff_heuristic == ELBOW_HEURISTIC_STR:
        return __elbow_index(distances_tuple, below_line_exclusive)
    elif cutoff_heuristic == DERIVATIVE_HEURISTIC_STR:
        return __shifted_derivative_index(
            sorted_values,
            distances_tuple=distances_tuple,
            below_line_exclusive=below_line_exclusive,
        )
    else:
        # we have an average
        elbow = __elbow_index(distances_tuple, below_line_exclusive)
        scores_index = __shifted_derivative_index(
            sorted_values,
            distances_tuple=distances_tuple,
            below_line_exclusive=below_line_exclusive,
        )
        return round((elbow + scores_index) / 2)


def find_elbow_index(
    values: ArrayLike,
    metric: str = __MANHATTAN_STR,
    first_elbow: bool = True,
    max_first_delta: Optional[float] = 0.33,
    below_line_exclusive: bool = True,
) -> int:
    """Finds the elbow index (inclusive) in a series of descending real values.

    Example uses include selecting the number of topics and determining
    when there is a jump in a distance metric.

    Parameters
    ----------
    values: ArrayLike
        A 1d array (or list) of real values.
    metric: str (Optional default "manhattan")
        Which distance metric to use when comparing with the line.
        One of ("euclidean", "manhattan", "uniform").
    first_elbow: bool (Optional default True)
        If true only the first elbow will be examined in the
        graph.
        Elbow finding can behave poorly compared to human intuition
        when values cross the comparison line in a sort of S-curve.
    max_first_delta_percent: Optional[float] (Optional default .33)
        Max value allowed for (y[1] - y[0]) / (y[0] - y[-1]).
        Some data sets have a single close value and
        then lots of bad ones and return unintuitive results
        as the distance from the line for index 0 is always 0.
        The default of 0.33 says that if 33% or more of the total
        change happens between 0 and 1 we will say the elbow is
        index 0.
    below_line_exclusive: bool (Optional default True)
        If true then result indices which are from an elbow below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    Returns
    -------
    int
        The index of the point with the greatest perpendicular
        distance from the comparison line when all provided data
        has been sorted in descending order.
        Will return -1 if provided a None value, empty list,
        or zero vector.

    Notes
    -----
    The comparison slope is created by assuming a linear descent
    from the largest to the smallest provided value. The elbow
    is the farthest perpendicular distance from the line.
    This can then be thought of as a value which diverges the most
    from a linear decrease and can then be used as a cut-off value.

    It is imposible to detect if anything diverges when given
    2 or fewer points, so index zero will be returned.
    """
    return find_cutoff(
        values,
        ELBOW_HEURISTIC_STR,
        distance_metric=metric,
        first_elbow=first_elbow,
        max_first_delta=max_first_delta,
        below_line_exclusive=below_line_exclusive,
    )


def __elbow_index(
    distances_tuple: LineDistances, below_line_exclusive: bool = True
) -> int:
    raw_elbow = distances_tuple.distances.argmax()
    if below_line_exclusive and distances_tuple.y_deltas[raw_elbow] < 0:
        return raw_elbow - 1
    else:
        return raw_elbow


def __shifted_derivative_index(
    sorted_values: ArrayLike,
    distances_tuple: LineDistances,
    below_line_exclusive: bool = True,
) -> int:
    """Finds a cutoff value based on distance from curve multiplied by
    shifted second derivative of values.

    This rewards points which are far from the curve which also have large
    changes in slope.

    Parameters
    ----------
    below_line_exclusive: bool (Optional default True)
        If true then result indices which are from below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.
    """
    # We want to have the 2nd derivative slid one to the left, that way it will have a high value at
    # the point where things change a lot
    slid_second_derivative = _get_shifted_second_derivative(
        sorted_values, distances_tuple.is_truncated, distances_tuple.truncation_index
    )
    # Now multiply these together and find the max
    scores = (
        distances_tuple.distances[: distances_tuple.truncation_index + 1]
        * slid_second_derivative
    )
    res_index = scores.argmax()
    if below_line_exclusive and distances_tuple.y_deltas[res_index] < 0:
        return res_index - 1
    else:
        return res_index


def _get_shifted_second_derivative(
    sorted_values: NDArray[np.float64], is_truncated: bool, truncation_index: int
):
    """Get the absolute value of the 2nd derivative slid one to the left so that it will have
    a high value at the point where things change a lot."""
    if truncation_index > sorted_values.size:
        raise ValueError("truncation_index must be less than sorted_values.size")
    if is_truncated:
        first_derivative = np.hstack(
            (
                [0],
                sorted_values[1 : truncation_index + 2]
                - sorted_values[: truncation_index + 1],
            )
        )
        second_derivative = np.abs(
            np.hstack(
                (
                    [0],
                    first_derivative[1 : truncation_index + 2]
                    - first_derivative[: truncation_index + 1],
                )
            )
        )
        # trim first and second derivative
        first_derivative = first_derivative[:-1]
        slid_second_derivative = second_derivative[1:]
    else:
        first_derivative = np.hstack(
            (
                [0],
                sorted_values[1:] - sorted_values[:-1],
            )
        )
        second_derivative = np.abs(
            np.hstack(([0], first_derivative[1:] - first_derivative[:-1]))
        )
        # trim second derivative
        slid_second_derivative = np.hstack((second_derivative[1:], [0]))
    return slid_second_derivative
