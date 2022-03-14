from typing import Optional, Dict, Tuple

import numpy as np
from numpy.typing import NDArray
from top2vec.cutoff_heuristics.cutoff_heuristics import (
    find_cutoff,
    get_distances_from_line,
    _get_shifted_second_derivative,
    ELBOW_HEURISTIC_STR,
    DERIVATIVE_HEURISTIC_STR,
    AVERAGE_HEURISTIC_STR,
    RECURSIVE_ELBOW_HEURISTIC_STR,
)

import matplotlib.pyplot as plt


def plot_heuristic(
    values: NDArray,
    figure_num: str = "1",
    figsize: Tuple[int, int] = (16, 8),
    cutoff_args: Optional[Dict] = None,
    print_elbows=True,
):
    """Displays the various cutoff heuristics as applied to a
    series of real values.

    Parameters
    ----------
    values: NDArray
        A numpy array of values to plot.

    figure_num: str (Optional, default "1")
        Allow overwriting the same figure for matplotlib.

    figsize: Tuple[int, int]
        Desired size of figure in inches

    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.cutoff_heurstics.find_cutoff`
        for more information.

        cutoff_heuristic: str (Optional, default `'elbow'`)
            Which cutoff heuristic to use.
            See `top2vec.cutoff_heuristics` for more.
        first_elbow: bool (Optional, default True)
            If the curve forms an S around the linear descent line only
            return a cutoff from the first portion above/below the line.
        max_first_delta: Optional[float] = 0.33
            Use index 0 as cutoff if this value is exceeded as percent of total
            variation.
            Due to the way that elbow finding works this returns unintuitive
            results if the first value is vastly different than all following values
            unless this is set.
        below_line_exclusive: bool (Optional, default True)
            Will cutoff indices below the linear descent line be
            treated as exclusive.
            Note that this will cause points to be drawn one x value less
            than the maximum values on the various plots.

    print_elbows: bool
        Print the elbow values that were found for the provided data.
    """
    if cutoff_args is not None:
        first_elbow = cutoff_args.get("first_elbow", True)
        below_line_exclusive = cutoff_args.get("below_line_exclusive", True)
        max_first_delta = cutoff_args.get("max_first_delta", 0.33)
    else:
        first_elbow = True
        below_line_exclusive = True
        max_first_delta = 0.33

    sorted_values = np.flip(np.sort(np.array(values)))
    x = np.arange(sorted_values.size)

    m = (sorted_values[-1] - sorted_values[0]) / (sorted_values.size - 1)
    line = x * m + sorted_values[0]

    elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=ELBOW_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
    )
    distances_tuple = get_distances_from_line(
        sorted_values, m, sorted_values[0], first_elbow=first_elbow
    )
    y_distances = distances_tuple.y_deltas

    slid_second_derivative = _get_shifted_second_derivative(
        sorted_values, distances_tuple.is_truncated, distances_tuple.truncation_index
    )
    scores = (
        distances_tuple.distances[: distances_tuple.truncation_index + 1]
        * slid_second_derivative
    )
    alt_elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=DERIVATIVE_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
    )
    average_elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=AVERAGE_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
    )
    recursive_elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=RECURSIVE_ELBOW_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
    )
    cutoff_indices = {
        ELBOW_HEURISTIC_STR: elbow,
        DERIVATIVE_HEURISTIC_STR: alt_elbow,
        AVERAGE_HEURISTIC_STR: average_elbow,
        RECURSIVE_ELBOW_HEURISTIC_STR: recursive_elbow,
    }
    if print_elbows:
        print(f"Cutoff indices: {cutoff_indices}")
    ELBOW_COLOR = "blue"
    DERIVATIVE_COLOR = "orange"
    AVERGE_COLOR = "green"
    RECURSIVE_COLOR = "purple"

    fig = plt.figure(num=figure_num, clear=True, figsize=figsize)

    gs = fig.add_gridspec(nrows=3, ncols=3)
    ax = fig.add_subplot(gs[:2, 0])
    ax.plot(line)
    ax.scatter([elbow], [sorted_values[elbow]], color=ELBOW_COLOR)
    ax.scatter([alt_elbow], [sorted_values[alt_elbow]], color=DERIVATIVE_COLOR)
    ax.scatter([average_elbow], [sorted_values[average_elbow]], color=AVERGE_COLOR)
    ax.scatter(
        [recursive_elbow], [sorted_values[recursive_elbow]], color=RECURSIVE_COLOR
    )
    ax.plot(sorted_values)

    ax_y = fig.add_subplot(gs[2, 0])
    ax_y.axhline(0, color="black")
    ax_y.plot(y_distances)
    ax_y.scatter([elbow], [y_distances[elbow]], color=ELBOW_COLOR)
    ax_y.scatter([alt_elbow], [y_distances[alt_elbow]], color=DERIVATIVE_COLOR)
    ax_y.scatter([average_elbow], [y_distances[average_elbow]], color=AVERGE_COLOR)
    ax_y.scatter(
        [recursive_elbow], [y_distances[recursive_elbow]], color=RECURSIVE_COLOR
    )

    ax_d = fig.add_subplot(gs[0, 1])
    ax_d.plot(distances_tuple.distances[: distances_tuple.truncation_index + 1])
    ax_d.axhline(0, color="black")
    ax_d.scatter([elbow], [distances_tuple.distances[elbow]], color=ELBOW_COLOR)
    ax_d.xaxis.set_ticklabels([])
    ax_val_second_d = fig.add_subplot(gs[1, 1])
    ax_val_second_d.plot(slid_second_derivative)
    ax_val_second_d.scatter(
        [alt_elbow], [slid_second_derivative[alt_elbow]], color=DERIVATIVE_COLOR
    )
    ax_val_second_d.axhline(0, color="black")
    ax_val_scores = fig.add_subplot(gs[2, 1])
    ax_val_scores.plot(scores)
    ax_val_scores.scatter([alt_elbow], [scores[alt_elbow]], color=DERIVATIVE_COLOR)
    ax_val_scores.axhline(0, color="black")

    if elbow > 1:
        recursive_axis = fig.add_subplot(gs[0, 2])
        recursive_axis.xaxis.set_ticklabels([])
        recursive_values = sorted_values[: elbow + 1]
        recursive_x = np.arange(recursive_values.size)

        recursive_m = (recursive_values[-1] - recursive_values[0]) / (
            recursive_values.size - 1
        )
        recursive_line = recursive_x * recursive_m + recursive_values[0]
        recursive_distances_tuple = get_distances_from_line(
            recursive_values, recursive_m, recursive_values[0], first_elbow=first_elbow
        )
        recursive_y_distances = recursive_distances_tuple.y_deltas
        recursive_axis.plot(recursive_line)
        recursive_axis.plot(recursive_values)
        recursive_axis.scatter(
            [recursive_elbow], recursive_values[recursive_elbow], color=RECURSIVE_COLOR
        )

        recursive_ax_y = fig.add_subplot(gs[1, 2])
        recursive_ax_y.plot(recursive_y_distances)
        recursive_ax_y.scatter(
            [recursive_elbow],
            recursive_y_distances[recursive_elbow],
            color=RECURSIVE_COLOR,
        )
