"""Allow finding an elbow for cut-off heuristics.

Will eventually be rolled directly into Top2Vec.

Author: Shawn
License: BSD 3 clause
"""
from typing import Optional
import numpy as np
from numpy.typing import NDArray, ArrayLike

# Prevent typos
__EUCLIDEAN_STR = "euclidean"
__MANHATTAN_STR = "manhattan"
__UNIFORM_STR = "uniform"


def find_elbow_index(values: ArrayLike, metric: str = __EUCLIDEAN_STR) -> Optional[int]:
    """Finds the "elbow" in a series of descending real values.

    Example uses include selecting the number of topics and determining
    when there is a jump in a distance metric.

    Parameters
    ----------
    values: ArrayLike
        A 1d array (or list) of real values.
    metric: str
        Which distance metric to use when comparing with the line.
        One of ("euclidean", "manhattan", "uniform").

    Returns
    -------
    int, optional
        The index of the farthest point from the comparison slope
        when all provided data has been sorted in descending order.
        Will return None if provided a None value or empty list.

    Notes
    -----
    The comparison slope is created by assuming a linear descent
    from the largest to the smallest provided value. The elbow
    can then be thought of as a value which diverges the most
    and can then be used as a cut-off value.
    """
    if values is None:
        return None
    # Make sure the provided values are a sorted numpy array
    sorted_values = -np.sort(-np.array(values))
    if sorted_values.size == 0:
        return None
    if sorted_values.size == 1:
        return 0
    if len(sorted_values.shape) != 1:
        raise ValueError("Elbow finding must be a 1-D Array")
    slope = (sorted_values[-1] - sorted_values[0]) / (sorted_values.size - 1)
    y_intercept = sorted_values[0]

    distances = get_distances_from_line(
        sorted_values, slope, y_intercept, metric=metric
    )
    return distances.argmax()


def __euclidean_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L2 distance between two points."""
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5


def __manhattan_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L1 distance between two points"""
    return abs(x1 - x2) + abs(y1 - y2)


def __uniform_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate the L-infinity distance between two points"""
    return max(abs(x1 - x2), abs(y1 - y2))


def get_distances_from_line(
    values: ArrayLike,
    comparison_slope: float,
    comparison_y_intercept: float,
    metric: str = __EUCLIDEAN_STR,
) -> NDArray[np.float64]:
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
    metric: str
        Which distance metric to use when comparing with the line.
        One of ("euclidean", "manhattan", "uniform").

    Returns
    -------
    NDArray[np.float64]
        A 1d array of the shortest euclidean distance for each
        value to the provided line.
    """
    # This is all 2d, so euclidean seems like it should be the best by default
    if metric == __EUCLIDEAN_STR:
        dist_fun = __euclidean_distance
    elif metric == __MANHATTAN_STR:
        dist_fun = __manhattan_distance
    elif metric == __UNIFORM_STR:
        dist_fun = __uniform_distance
    else:
        raise ValueError(
            f"Illegal metric - '{metric}'.\
           Must be one of [{__EUCLIDEAN_STR}, {__MANHATTAN_STR}, {__UNIFORM_STR}]"
        )
    try:
        distances = np.zeros(values.size)
        to_examine = range(values.size)
    except AttributeError:
        # handed a list rather than a numpy array
        distances = np.zeros(len(values))
        to_examine = range(len(values))
    perp_slope = comparison_slope * -1
    # only compute this once
    divisor = comparison_slope - perp_slope
    # TODO: look at np.vectorize
    for x in to_examine:
        instance_y = values[x]
        # special case: slope of 0
        if divisor == 0:
            distances[x] = abs(instance_y - comparison_y_intercept)
        else:
            # Which y value are we computing distance for
            instance_y_intercept = instance_y - perp_slope * x

            comparison_x = (instance_y_intercept - comparison_y_intercept) / divisor
            comparison_y = comparison_slope * comparison_x + comparison_y_intercept

            distances[x] = dist_fun(x, instance_y, comparison_x, comparison_y)
    return np.array(distances)
