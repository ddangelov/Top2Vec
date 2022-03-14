"""Various methods to visualize various cutoff heuristics.

Author: Shawn
License: BSD 3 clause
"""
from typing import Optional, Dict, Tuple, Sequence, List

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
from top2vec.types import DocumentId, RetrievedDocuments, SimilarItems
from top2vec.Top2Vec import Top2Vec

import sklearn.metrics
import matplotlib.pyplot as plt


def plot_heuristic(
    values: NDArray,
    figure_num: str = "1",
    figsize: Tuple[int, int] = (16, 8),
    cutoff_args: Optional[Dict] = None,
    print_elbows: bool = True,
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
        min_for_elbow_recurse: int (Optional default 10)
            The minimum index for recursing if using `recursive_elbow`.
            If the elbow index for the first pass is less than this
            value it will be returned.
            Generally allows for better results if the first elbow
            is a small value such as due to a sudden large drop.

    print_elbows: bool
        Print the elbow values that were found for the provided data.
    """
    if cutoff_args is not None:
        first_elbow = cutoff_args.get("first_elbow", True)
        below_line_exclusive = cutoff_args.get("below_line_exclusive", True)
        max_first_delta = cutoff_args.get("max_first_delta", 0.33)
        min_for_elbow_recurse = cutoff_args.get("max_for_elbow_recurs", 10)
    else:
        first_elbow = True
        below_line_exclusive = True
        max_first_delta = 0.33
        min_for_elbow_recurse = 10

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
        min_for_elbow_recurse=min_for_elbow_recurse,
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
        min_for_elbow_recurse=min_for_elbow_recurse,
    )
    average_elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=AVERAGE_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
        min_for_elbow_recurse=min_for_elbow_recurse,
    )
    recursive_elbow = find_cutoff(
        sorted_values,
        cutoff_heuristic=RECURSIVE_ELBOW_HEURISTIC_STR,
        first_elbow=first_elbow,
        below_line_exclusive=below_line_exclusive,
        max_first_delta=max_first_delta,
        min_for_elbow_recurse=min_for_elbow_recurse,
    )
    cutoff_indices = {
        ELBOW_HEURISTIC_STR: elbow,
        DERIVATIVE_HEURISTIC_STR: alt_elbow,
        AVERAGE_HEURISTIC_STR: average_elbow,
        RECURSIVE_ELBOW_HEURISTIC_STR: recursive_elbow,
    }
    if print_elbows:
        print(f"Cutoff indices (inclusive): {cutoff_indices}")
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


# Some wrapper functions to also show how the heuristic is making decisions
def get_and_plot_similar_words(
    top2vec_model: Top2Vec,
    keywords: Optional[Sequence[str]] = None,
    keywords_neg: Optional[Sequence[str]] = None,
    max_returned: Optional[int] = 250,
    new_cutoff_args: Optional[Dict] = None,
    figure: Optional[str] = None,
    print_first: Optional[int] = 20,
) -> SimilarItems:
    """Find similar words and optionally plot the cutoff heuristics.

    Parameters
    ----------
    top2vec_model: Top2Vec
        A trained model to query.
    keywords: Sequence[str]
        List of positive keywords being used for search of semantically
        similar words.
    keywords_neg: Optional[Sequence[str]] (Optional, default None)
        List of negative keywords being used for search of semantically
        dissimilar words.
    max_returned: Optional[int] (Optional, default 250)
        Maximum number of words to return.
    new_cutoff_args: Optional[Dict] (Optional, default None)
        Temporarily change the cutoff_args of top2vec_model
        when determining this cutoff index.
    figure: Optional[str] (Optional, default None)
        Provide a figure name for plotting heuristics.
        Leave None to not plot anything.
    print_first: Optaional[int] (Optional, default 20)
        Will print out the top `print_first` cosine similarities to screen.

    Notes
    -----
    The cosine similarity to the provided keywords (and keywords_neg) will
    be ignored for purposes of determining a cutoff if `use_cutoff_heuristics`.

    Returns
    -------
    SimilarItems
        Tuple index 0 is the words in an NDArray, most similar first.
        Tuple index 1 is the cosine similarity of the words and average
        of keyword vectors as an NDArray.
    """
    if new_cutoff_args is not None:
        old_cutoff_args = top2vec_model.cutoff_args
        top2vec_model.cutoff_args = new_cutoff_args
    similar_words, similar_scores = top2vec_model.similar_words(
        keywords, max_returned, keywords_neg=keywords_neg, use_index=False
    )
    if figure is not None:
        if keywords_neg is None:
            _keywords_neg = []
        else:
            _keywords_neg = keywords_neg
        combined_vector = top2vec_model._get_combined_vec(
            vecs=top2vec_model._words2word_vectors(keywords),
            vecs_neg=top2vec_model._words2word_vectors(_keywords_neg),
        )
        cosine_sims = (
            1
            - sklearn.metrics.pairwise.pairwise_distances(
                [combined_vector],
                top2vec_model.word_vectors,
                metric="cosine",
            )[0]
        )
        # Need to remove the ignored items
        ignore_indices_array = np.hstack(
            (
                np.array([top2vec_model.word_indexes[word] for word in keywords]),
                np.array([top2vec_model.word_indexes[word] for word in _keywords_neg]),
            )
        )
        fancy_indices = np.setdiff1d(
            np.arange(cosine_sims.shape[0]), ignore_indices_array
        )
        sorted_cosine_sims = -np.sort(-cosine_sims[fancy_indices])
        if print_first:
            print(
                "Raw cosine similarities (minus ignore): ",
                sorted_cosine_sims[:print_first],
            )
        plot_heuristic(
            sorted_cosine_sims, figure_num=figure, cutoff_args=top2vec_model.cutoff_args
        )
    if new_cutoff_args is not None:
        top2vec_model.cutoff_args = old_cutoff_args

    return SimilarItems(similar_words, similar_scores)


def get_and_plot_topic_description(
    top2vec_model: Top2Vec,
    topic_num: int,
    reduced: bool = False,
    max_returned: Optional[int] = 250,
    new_cutoff_args: Optional[Dict] = None,
    figure: Optional[str] = None,
    print_first: Optional[int] = 20,
) -> SimilarItems:
    """Find descriptive words for a topic and optionally plot the
    cutoff heuristics.

    Parameters
    ----------
    top2vec_model: Top2Vec
        A trained model to query.
    topic_num: int
        The topic number to query.
    reduced: bool
        Whether the topic number should be from the
        hierarchically reduced models.
    max_returned: Optional[int] (Optional, default 250)
        Maximum number of words to return.
    new_cutoff_args: Optional[Dict] (Optional, default None)
        Temporarily change the cutoff_args of top2vec_model
        when determining this cutoff index.
    figure: Optional[str] (Optional, default None)
        Provide a figure name for plotting heuristics.
        Leave None to not plot anything.
    print_first: Optaional[int] (Optional, default 20)
        Will print out the top `print_first` cosine similarities to screen.

    Returns
    -------
    SimilarItems
        Tuple index 0 is the words in an NDArray, most similar first.
        Tuple index 1 is the cosine similarity of the words and average
        of keyword vectors as an NDArray.
    """
    if new_cutoff_args is not None:
        old_cutoff_args = top2vec_model.cutoff_args
        top2vec_model.cutoff_args = new_cutoff_args
    if reduced:
        t_vector = top2vec_model.topic_vectors_reduced[topic_num]
    else:
        t_vector = top2vec_model.topic_vectors[topic_num]
    similar_words, similar_scores = top2vec_model.search_words_by_vector_heuristic(
        t_vector, max_returned
    )
    if figure is not None:
        cosine_sims = 1 - sklearn.metrics.pairwise.pairwise_distances(
            [t_vector], top2vec_model.word_vectors, metric="cosine"
        )
        sorted_cosine_sims = -np.sort(-cosine_sims[0])
        if print_first:
            print("Raw cosine similarities: ", sorted_cosine_sims[:print_first])
        plot_heuristic(
            sorted_cosine_sims, figure_num=figure, cutoff_args=top2vec_model.cutoff_args
        )
    if new_cutoff_args is not None:
        top2vec_model.cutoff_args = old_cutoff_args

    return SimilarItems(similar_words, similar_scores)


def get_and_plot_similar_documents(
    top2vec_model: Top2Vec,
    doc_ids: Sequence[DocumentId],
    doc_ids_neg: Optional[Sequence[DocumentId]] = None,
    max_returned: Optional[int] = 250,
    new_cutoff_args: Optional[Dict] = None,
    figure: Optional[str] = None,
    print_first: Optional[int] = 20,
) -> RetrievedDocuments:
    """Find descriptive words for a topic and optionally plot the
    cutoff heuristics.

    Parameters
    ----------
    top2vec_model: Top2Vec
        A trained model to query.
    doc_ids: Sequence[DocumentId]
        Unique ids of document. If ids were not given, the index of
        document in the original corpus.

    doc_ids_neg: Optional[Sequence[DocumentId]] (Optional default None)
        Unique ids of document. If ids were not given, the index of
        document in the original corpus.
    max_returned: Optional[int] (Optional, default 250)
        Maximum number of documents to return.
    new_cutoff_args: Optional[Dict] (Optional, default None)
        Temporarily change the cutoff_args of top2vec_model
        when determining this cutoff index.
    figure: Optional[str] (Optional, default None)
        Provide a figure name for plotting heuristics.
        Leave None to not plot anything.
    print_first: Optaional[int] (Optional, default 20)
        Will print out the top `print_first` cosine similarities to screen.

    Returns
    -------
    RetrievedDocuments
        The documents most similar to the average of the provided
        document vectors.
        (Documents, Cosine Similarities, Doc_Ids)
        Tuple index 0 will be None if `return_documents` is False or
        `Top2Vec.documents` is None.
    """
    if new_cutoff_args is not None:
        old_cutoff_args = top2vec_model.cutoff_args
        top2vec_model.cutoff_args = new_cutoff_args

    retrieved_docs = top2vec_model.search_documents_by_documents(
        doc_ids=doc_ids,
        doc_ids_neg=doc_ids_neg,
        num_docs=max_returned,
        return_documents=True,
    )

    if figure is not None:
        if doc_ids_neg is None:
            _doc_ids_neg = []
        else:
            _doc_ids_neg = doc_ids_neg
        doc_indexes = top2vec_model._get_document_indexes(doc_ids)
        doc_indexes_neg = top2vec_model._get_document_indexes(_doc_ids_neg)
        combined_vector = top2vec_model._get_combined_vec(
            vecs=[top2vec_model.document_vectors[ind] for ind in doc_indexes],
            vecs_neg=[top2vec_model.document_vectors[ind] for ind in doc_indexes_neg],
        )

        cosine_sims = (
            1
            - sklearn.metrics.pairwise.pairwise_distances(
                [combined_vector], top2vec_model.document_vectors, metric="cosine"
            )[0]
        )
        # Need to remove the ignored items
        ignore_indices_array = np.hstack((doc_indexes, doc_indexes_neg))
        fancy_indices = np.setdiff1d(
            np.arange(cosine_sims.shape[0]), ignore_indices_array
        )
        sorted_cosine_sims = -np.sort(-cosine_sims[fancy_indices])
        if print_first:
            print(
                "Raw cosine similarities (minus ignore): ",
                sorted_cosine_sims[:print_first],
            )
        plot_heuristic(
            sorted_cosine_sims, figure_num=figure, cutoff_args=top2vec_model.cutoff_args
        )

    if new_cutoff_args is not None:
        top2vec_model.cutoff_args = old_cutoff_args
    return retrieved_docs
