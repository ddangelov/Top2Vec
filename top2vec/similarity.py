"""Determine what is and isn't similar between sets of vectors using the elbow finding heuristic."""
from typing import Tuple, List, Optional

import sklearn.metrics
import numpy as np
from numpy.typing import NDArray, ArrayLike

from top2vec.elbow_finding import find_elbow_index


def find_closest_items(
    comparison_vectors: ArrayLike,
    embedding: NDArray,
    elbow_metric: str = "euclidean",
    maxN: Optional[int] = None,
) -> List[Tuple[NDArray[np.int64], NDArray[np.float64]]]:
    """Finds the closest embeddings based on provided vector(s) from the same space.

    vectors: ArrayLike
        Something which can be interpreted as a 1D or 2D numpy array of
        floats. Will be used as the points to compute distance from.
    comparison_embedding: NDarray
        A 2D numpy array of floats.
        Will be compared to vectors and saved if "close enough".
    maxN: Optional[int]
        A maximum number of points to consider similar based on the
        elbow finding heuristic. The number of returned similarity
        scores will be the minimum of the elbow cut-off and maxN
        if provided.
    elbow_metric: str
        Which distance metric to use when computing the cut-off for
        close enough.

    Returns
    -------
    NDArray[np.float64]
        A list of tuples where index 0 is a numpy array of the indices of similar vectors and
        index 1 is a numpy array of their cosine similarity scores.
        Tuple i will correspond to the provided comparison_vectors i.
    """
    try:
        if len(comparison_vectors.shape) == 1:
            vectors = comparison_vectors.reshape(1, -1)
        else:
            vectors = comparison_vectors
    except AttributeError:
        # Provided a list rather than an array
        tmp_vectors = np.array(comparison_vectors)
        if len(tmp_vectors.shape) == 1:
            vectors = tmp_vectors.reshape(1, -1)
        else:
            vectors = tmp_vectors
    if len(embedding.shape) != 2:
        raise ValueError(
            f"Embedding must be 2D Array, provided {len(embedding.shape)}D"
        )
    # This parallelizes nicely
    similarity_scores = 1 - sklearn.metrics.pairwise_distances(
        vectors, embedding, metric="cosine"
    )
    # Need to broadcast this for each if we are multiple vectors at once
    elbow_indices = np.apply_along_axis(
        find_elbow_index, arr=similarity_scores, axis=1, metric=elbow_metric
    )
    """
        # Turns out that I'm iterating through anyways, so I don't need to do this
        if maxN is not None:
            # Is there a better way to do this?
            vmin = np.vectorize(min)
            cutoff_indices = vmin(elbow_indices, maxN)
        else:
            cutoff_indices = elbow_indices
    """
    relevant_indices = np.flip(np.argsort(similarity_scores), axis=1)
    # Now I reshape the individual vectors
    # NumPy doesn't support jagged arrays, so now is time to iterate
    result = []
    for row in range(relevant_indices.shape[0]):
        if maxN is not None:
            cutoff = min(elbow_indices[row], maxN)
        else:
            cutoff = elbow_indices[row]
        item_indices = relevant_indices[row, :cutoff]
        item_scores = similarity_scores[row, item_indices]
        result.append((item_indices, item_scores))
    return result


def describe_topics(
    topic_vectors: NDArray[np.float64],
    vocabulary_embedding: NDArray[np.float64],
    vocabulary: ArrayLike,
    maxTerms: int = 100,
    elbow_metric: str = "euclidean",
) -> List[Tuple[NDArray, NDArray[np.float64]]]:
    """Finds the most descriptive terms for a topic or set of topics using cosine
    similarity and an elbow finding heuristic.

    Raises
    ------
    ValueError if the vocabulary length is not equal to the 1st dimension of the embedding.

    Notes
    -----
    This will be much more efficient if vocabulary reduction has already been performed.
    """
    # TODO: Generecize this function description. This can describe topics but it can also
    # be used to describe document to topic mappings or even document term mappings
    try:
        vocab_len = vocabulary.shape[0]
        vocab_array = vocabulary
    except AttributeError:
        # we have a list
        vocab_len = len(vocabulary)
        vocab_array = np.array(vocabulary)
    if vocab_len != vocabulary_embedding.shape[0]:
        raise ValueError(
            f"Vocabulary size ({len(vocabulary)}) != vocabulary embedding size ({vocabulary_embedding.shape[0]})"
        )
    closest_terms = find_closest_items(
        topic_vectors, vocabulary_embedding, maxN=maxTerms, elbow_metric=elbow_metric
    )
    results = []
    for indices, scores in closest_terms:
        results.append((vocab_array[indices], scores))
    return results


def generate_similarity_matrix(
    vectors: ArrayLike,
    comparison_embeddings: ArrayLike,
    maxN: int = 100,
    elbow_metric: str = "euclidean",
) -> NDArray[np.float64]:
    """Translates from a series of vectors and a set of embeddings to compare
    into a matrix. Uses the elbow finding heuristic to determine what is
    "similar enough" to keep.

    Providing a set of topic vectors and the corresponding term embeddings
    will generate a topic x term matrix which is used in other NLP algorithms.
    Similarly, providing a set of document vectors and the corresponding
    topic vectors in the same space will generate a document x topic matrix.

    Parameters
    ----------
    vectors: ArrayLike
        Something which can be interpreted as a 1D or 2D numpy array of
        floats. Will be used as the points to compute distance from.
    comparison_embedding: ArrayLike
        Something which can be interpreted as a 2D numpy array of floats.
        Will be compared to vectors and saved if "close enough".
    maxN: int
        A maximum number of points to consider similar to a provided
        vector. Can be used to limit topic sizes.
    elbow_metric: str
        Which distance metric to use when computing the cut-off for
        close enough.

    Returns
    -------
    NDArray[np.float64]
        A matrix with N rows and M columns.
        N is equal to the number of provided vectors and M is equal
        to the number of provided embeddings.
        The ith row will have values of the cosine similarity for
        all columns [j0, j1, ..., jn] which were deemed to be
        close enough to vector i based on the elbow finding heuristic.
    """
    # TODO: Give option to make a sparse matrix
    try:
        vector_array = vectors
        num_vectors = vectors.shape[0]
    except AttributeError:
        vector_array = np.array(vectors)
        num_vectors = vector_array.shape[0]
    try:
        embeddings_array = comparison_embeddings
        num_embeddings = embeddings_array.shape[0]
    except AttributeError:
        embeddings_array = np.array(comparison_embeddings)
        num_embeddings = embeddings_array.shape[0]
    similarity_values = find_closest_items(
        vector_array, embeddings_array, maxN=maxN, elbow_metric=elbow_metric
    )
    res_matrix = np.zeros((num_vectors, num_embeddings))
    for index, (indices, scores) in enumerate(similarity_values):
        res_matrix[index][indices] = scores
    return res_matrix
