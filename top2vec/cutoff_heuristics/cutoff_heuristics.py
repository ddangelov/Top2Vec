"""Supports various heuristics for determining when values deviate from an expected line.

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
* If `first_elbow` is true and the curve crosses the linear descent line
  then only the segment of the curve prior to the first crossing will
  be examined. This will not cause a re-calculation of the linear
  descent slope.

From here the individual heuristics are run

* `elbow`: Finds the index with the greatest distance from the curve.
This is the 'standard' heuristic for things such as determining the number
of topics to represent a corpus of documents.

* `shifted_derivative`: Finds the index with the greatest
`distance[i] * 2nd_derivative[i + 1]`.
This prioritizes points which have a large change in slope but
can under-estimate if the curve is a long slow decay.

* `average`: Runs both of the above and returns the average index
between the two (rounding up).

* `recursive_elbow`: As `elbow`, but runs twice if there are at
least min_for_elbow_recurse points to examine in the sub-graph.
The first pass is used as a cutoff which is then passed back into
the elbow finding method. Seems to behave better when the data
forms a long slow curve.
"""

from typing import Optional
import numpy as np
from numpy.typing import NDArray, ArrayLike
from top2vec.types import LineDistances


ELBOW_HEURISTIC_STR = "elbow"
DERIVATIVE_HEURISTIC_STR = "shifted_derivative"
AVERAGE_HEURISTIC_STR = "average"
RECURSIVE_ELBOW_HEURISTIC_STR = "recursive_elbow"
SUPPORTED_HEURISTICS = [
    ELBOW_HEURISTIC_STR,
    DERIVATIVE_HEURISTIC_STR,
    AVERAGE_HEURISTIC_STR,
    RECURSIVE_ELBOW_HEURISTIC_STR,
]


def get_distances_from_line(
    sorted_values: ArrayLike,
    comparison_slope: float,
    comparison_y_intercept: float,
    first_elbow: bool = True,
) -> LineDistances:
    """Finds the distance for all provided values from a provided line.

    Parameters
    ----------
    values : ArrayLike
        A 1d array (or list) of y values such that index 0 is y(0).

    comparison_slope : float
        The slope of the line to compare values with.

    comparison_y_intercept : float
        The y intercept of the line to compare values with.

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

    Notes
    -----
    The original version of this algorithm found the shortest
    distance between each point and the line via the perpendicular
    slope in order to handle if the shortest distancce was between
    two points that existed on the line.
    After running multiple situations I haven't found a case
    where simply subtracting values[i] from line[i] doesn't result
    in the same index as the greatest distance from the line.
    Therefore in order to save computational cost the algorithm
    has been greatly simplified.
    """
    try:
        n_elements = sorted_values.size
        sorted_values_array = sorted_values
    except AttributeError:
        # handed a list rather than a numpy array
        n_elements = len(sorted_values)
        sorted_values_array = np.array(sorted_values)

    comparison_line = comparison_slope * np.arange(n_elements) + comparison_y_intercept
    differences = sorted_values_array - comparison_line
    abs_differences = np.abs(differences)

    truncation_index = abs_differences.size - 1
    was_positive_y = None

    # Check that there is actually a value, then use argmax
    # to avoid making multiple arrays for this
    if n_elements > 0:
        max_val = np.max(differences)
        min_val = np.min(differences)
        if max_val > 0 and min_val < 0:
            # We have a flip
            first_positive_index = np.argmax(differences > 0)
            first_negative_index = np.argmax(differences < 0)
            if first_positive_index < first_negative_index:
                was_positive_y = True
                if first_elbow:
                    truncation_index = first_negative_index - 1
            else:
                was_positive_y = False
                if first_elbow:
                    truncation_index = first_positive_index - 1
            if first_elbow:
                differences[truncation_index + 1 :] = 0
                abs_differences[truncation_index + 1 :] = 0
        elif max_val > 0:
            # We default to this but I feel it is better to be explicit
            was_positive_y = True
        elif min_val < 0:
            was_positive_y = False

    return LineDistances(
        abs_differences,
        differences,
        truncation_index != n_elements - 1,
        truncation_index,
        was_positive_y is None or was_positive_y,
    )


def __edge_cases(sorted_values: ArrayLike, max_first_delta: Optional[float] = 0.33):
    """Handle edge cases prior to finding cut-off index"""
    if len(sorted_values.shape) != 1:
        raise ValueError("Values must be a 1-D Array")
    # The count_nonzero function is actually slow
    # Converting to a set and checking for equal to 0 is much slower
    if sorted_values.size == 0 or np.count_nonzero(sorted_values) == 0:
        return -1
    elif sorted_values.size <= 2:
        return 0
    rise = sorted_values[-1] - sorted_values[0]
    if max_first_delta is not None and rise != 0:
        # determine percent of total drop contained between bin 0 and 1
        percent_delta = (sorted_values[1] - sorted_values[0]) / rise
        if percent_delta >= max_first_delta:
            return 0
    return None


def find_cutoff(
    values: ArrayLike,
    cutoff_heuristic: str = RECURSIVE_ELBOW_HEURISTIC_STR,
    first_elbow: bool = True,
    max_first_delta: Optional[float] = 0.33,
    below_line_exclusive: bool = True,
    min_for_elbow_recurse: int = 10,
    already_sorted: bool = False,
):
    """Finds the cutoff index (inclusive) in a series of real values.

    Example uses include selecting the number of topics and determining
    when there is a jump in a similarity metric.

    Parameters
    ----------
    values: ArrayLike
        A 1d array (or list) of real values.

    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which heuristic to use when determining the index.

            * `elbow`: Finds the index with the greatest distance from the curve.
            * `shifted_derivative`: Finds the index with the greatest
            `distance[i] * 2nd_derivative[i + 1]`.
                This prioritizes points which have a large change in slope but
                can under-estimate if the curve is a long slow decay.
            * `average`: Runs both of the above and returns the average index
            between the two (rounding up).
            * `recursive_elbow`: As `elbow`, but runs twice.
             Seems to behave better when the data forms a long slow curve.

    first_elbow: bool (Optional, default True)
        If true only the first elbow will be examined in the
        graph.
        Cutoff finding can behave poorly compared to human intuition
        when values cross the comparison line in a sort of S-curve.

    max_first_delta_percent: Optional[float] (Optional, default .33)
        Max value allowed for `(y[1] - y[0]) / (y[0] - y[-1])`.
        Some data sets have a single close value and
        then lots of bad ones and return unintuitive results
        as the distance from the line for index 0 is always 0.
        The default of 0.33 says that if 33% or more of the total
        change happens between 0 and 1 we will say the elbow is
        index 0.
        Providing `None` causes this check to be skipped entirely.

    below_line_exclusive: bool (Optional, default True)
        If true then result indices which are from a cutoff below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    min_for_elbow_recurse: int (Optional, default 10)
        The minimum index for recursing if using `recursive_elbow`.
        If the elbow index for the first pass is less than this
        value it will be returned.
        Generally allows for better results if the first elbow
        is a small value such as due to a sudden large drop.

    already_sorted: bool (Optional, default False)
        Don't re-sort the data in descending order.

    Returns
    -------
    int
        The index of the point to be used as a cutoff when all
        provided data has been sorted in descending order, as
        determined by `cutoff_heuristic`.

        Will return -1 if provided a None value, empty list,
        or zero vector.

        It is imposible to detect if anything diverges when given
        2 or fewer points, so index 0 will be returned.

    Notes
    -----
    The comparison slope is created by assuming a linear descent
    from the largest to the smallest provided value. The elbow
    is the farthest distance from the line.
    This can then be thought of as a value which diverges the most
    from a linear decrease and can then be used as a cut-off value.
    """

    if cutoff_heuristic not in SUPPORTED_HEURISTICS:
        raise ValueError(
            f"Unsupported heuristic selected: {cutoff_heuristic}. Must be one of {SUPPORTED_HEURISTICS}."
        )
    if values is None:
        return -1

    # Make sure the provided values are a sorted numpy array
    if already_sorted:
        sorted_values = values
    else:
        sorted_values = np.flip(np.sort(np.array(values)))
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
    elif cutoff_heuristic == AVERAGE_HEURISTIC_STR:
        # we have an average
        elbow = __elbow_index(distances_tuple, below_line_exclusive)
        scores_index = __shifted_derivative_index(
            sorted_values,
            distances_tuple=distances_tuple,
            below_line_exclusive=below_line_exclusive,
        )
        return round((elbow + scores_index) / 2)
    elif cutoff_heuristic == RECURSIVE_ELBOW_HEURISTIC_STR:
        first_pass = __elbow_index(distances_tuple, below_line_exclusive)
        if first_pass >= min_for_elbow_recurse:
            return find_cutoff(
                sorted_values[: first_pass + 1],
                cutoff_heuristic=ELBOW_HEURISTIC_STR,
                first_elbow=first_elbow,
                max_first_delta=max_first_delta,
                below_line_exclusive=below_line_exclusive,
                already_sorted=True,
            )
        else:
            return first_pass


def find_elbow_index(
    values: ArrayLike,
    first_elbow: bool = True,
    max_first_delta: Optional[float] = 0.33,
    below_line_exclusive: bool = True,
    already_sorted: bool = False,
) -> int:
    """Finds the elbow index (inclusive) in a series of real values.

    Example uses include selecting the number of topics and determining
    when there is a jump in a similarity metric.

    Parameters
    ----------
    values: ArrayLike
        A 1d array (or list) of real values.
    first_elbow: bool (Optional, default True)
        If true only the first elbow will be examined in the
        graph.
        Cutoff finding can behave poorly compared to human intuition
        when values cross the comparison line in a sort of S-curve.

    max_first_delta_percent: Optional[float] (Optional, default .33)
        Max value allowed for `(y[1] - y[0]) / (y[0] - y[-1])`.
        Some data sets have a single close value and
        then lots of bad ones and return unintuitive results
        as the distance from the line for index 0 is always 0.
        The default of 0.33 says that if 33% or more of the total
        change happens between 0 and 1 we will say the elbow is
        index 0.
        Providing `None` causes this check to be skipped entirely.

    below_line_exclusive: bool (Optional, default True)
        If true then result indices which are from a cutoff below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    already_sorted: bool (Optional, default False)
        Don't re-sort the data in descending order.

    Returns
    -------
    int
        The index of the point with the greatest distance from the
        comparison line when all provided data has been sorted in
        descending order.
        Will return -1 if provided a None value, empty list,
        or zero vector.

        It is imposible to detect if anything diverges when given
        2 or fewer points, so index zero will be returned.

    Notes
    -----
    The comparison slope is created by assuming a linear descent
    from the largest to the smallest provided value. The elbow
    is the farthest distance from the line.
    This can then be thought of as a value which diverges the most
    from a linear decrease and can then be used as a cut-off value.
    """
    return find_cutoff(
        values,
        ELBOW_HEURISTIC_STR,
        first_elbow=first_elbow,
        max_first_delta=max_first_delta,
        below_line_exclusive=below_line_exclusive,
        already_sorted=already_sorted,
    )


def __elbow_index(
    distances_tuple: LineDistances, below_line_exclusive: bool = True
) -> int:
    """Find the index of the point farthest from the line.

    Parameters
    ----------
    distances_tuple: LineDistances
        The result of `get_distances_from_line`.

    below_line_exclusive: bool (Optional, default True)
        If true then result indices which are from below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    Returns
    -------
    int
        The index to be used as a cutoff (inclusive).
    """
    raw_elbow = distances_tuple.distances.argmax()
    if below_line_exclusive and distances_tuple.y_deltas[raw_elbow] < 0:
        return raw_elbow - 1
    else:
        return raw_elbow


def __shifted_derivative_index(
    sorted_values: NDArray[np.float64],
    distances_tuple: LineDistances,
    below_line_exclusive: bool = True,
) -> int:
    """Finds a cutoff value based on distance from curve multiplied by
    shifted second derivative of values.

    This rewards points which are far from the curve which also have large
    changes in slope.

    Parameters
    ----------
    sorted_values: NDArray[np.float64]
        A sorted (reverse order) array of real values to be used when
        computing the derivative.

    distances_tuple: LineDistances
        The result of `get_distances_from_line`.

    below_line_exclusive: bool (Optional, default True)
        If true then result indices which are from below
        the linear descent line will be treated as exclusive.
        Therefore the final result will be index - 1.

    Returns
    -------
    int
        The cutoff index (inclusive).

    Notes
    -----
    The 2nd derivative is slid one to the left so that there will be a high
    value at the point where things change a lot.
    """
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
    if truncation_index >= sorted_values.size:
        raise ValueError("truncation_index must be less than sorted_values.size")
    if is_truncated and truncation_index < sorted_values.size - 1:
        sub_array = sorted_values[: truncation_index + 2]
        second_derivative = np.abs(np.diff(sub_array, n=2, prepend=[0, 0]))
        return second_derivative[1:]
    else:
        second_derivative = np.abs(
            np.diff(sorted_values, n=2, prepend=[0, 0], append=0)
        )
        second_derivative[-1] = 0
        return second_derivative[1:]
