"""Determine what is and isn't similar between sets of vectors using the elbow finding heuristic."""
from typing import Tuple, List, Optional, Dict, NamedTuple

import sklearn.metrics
import numpy as np
from numpy.typing import NDArray, ArrayLike
import scipy.sparse
from top2vec.cutoff_heuristics import ELBOW_HEURISTIC_STR, find_cutoff

# TODO: Run profiling and think if there is a better way to
# do the distance computation


class VectorSimilarityScores(NamedTuple):
    """Represents multiple data points about distance from a line.

    Attributes
    ----------
    indices: NdArray[np.int64]
        An array of vector indices sorted from most to least similar.
    scores: NdArray[np.float64]
        Index 0 is the similarity score of `indices[0]` and the
        original vector.
    """

    indices: NDArray[np.int64]
    scores: NDArray[np.float64]


def __ensure_np_array(vectors: ArrayLike) -> Tuple[int, NDArray]:
    """Translate an ArrayLike into a numpy array (if necessary)
    and determine how many elements it has.
    """
    if vectors is None:
        return 0, None
    try:
        vector_array = vectors
        num_vectors = vectors.shape[0]
    except AttributeError:
        vector_array = np.array(vectors)
        num_vectors = vector_array.shape[0]
    return num_vectors, vector_array


def __ensure_2d_np_array(vectors: ArrayLike) -> Tuple[int, NDArray]:
    """Translate an ArrayLike into a numpy array (if necessary)
    and determine how many rows it has.
    """
    num_rows, vector_array = __ensure_np_array(vectors)
    # Special case: an empty 2D array is treated as having one row
    if vector_array is not None:
        if vector_array.size == 0:
            return 0, np.array([[]])
        elif len(vector_array.shape) == 1:
            return 1, vector_array.reshape(1, -1)
    return num_rows, vector_array


def find_closest_items(
    comparison_vectors: ArrayLike,
    comparison_embedding: NDArray,
    topn: Optional[int] = None,
    ignore_indices: Optional[ArrayLike] = None,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
) -> List[VectorSimilarityScores]:
    """Finds the closest embeddings based on provided vector(s) from the same space.

        Parameters
        ----------
        vectors: ArrayLike
            Something which can be interpreted as a 1D or 2D numpy array of
            floats. Will be used as the points to compute distance from.
        comparison_embedding: NDarray
            A 2D numpy array of floats.
            Will be compared to vectors and saved if "close enough".
        topn: Optional[int]
            A maximum number of points to consider similar based on the
            elbow finding heuristic. The number of returned similarity
            scores will be the minimum of the elbow cut-off and topn
            if provided.
        ignore_indices: Optional[ArrayLike]
            An array-like structure of indices to ignore when computing
            the elbow-threshold as well as for return values.
            Prevents you from getting the same thing out that you put in
            if comparison_vectors and embedding are the same data.
        require_positive: bool (Optional, default True)
            It is possibe to have bad data where there is only one point
            which is actually similar and the rest are orthogonal or worse.
            The distance from the line for that first point will be 0,
            which won't be an elbow.
            If True then only values which are greater than 0 will be returned.
        cutoff_heuristic: str (Optional, default `'elbow'`)
            Which cutoff heuristic to use.
            See `top2vec.cutoff_heuristics` for more.
        cutoff_args: dict (Optional, default None)
            Pass custom arguments to the cutoff heuristic.
            See `top2vec.cutoff_heurstics.find_cutoff` for more information.

            elbow_metric: str (Optional, default `'manhattan'`)
                Which distance metric to use when computing the cut-off for
                close enough.
            first_elbow: bool (Optional, default True)
                If the curve forms an S around the linear descent line only
                return an elbow from the first portion above/below the line.
            max_first_delta: Optional[float] = 0.33
                Use index 0 as elbow if this value is exceeded as percent of total
                variation.
                Due to the way that elbow finding works this returns unintuitive
                results if the first value is vastly different than all following values
                unless this is set.
            below_line_exclusive: bool (Optiona, default True)
                Will cutoff indices below the linear descent line be
                treated as exclusive.
    ]
        Returns
        -------
        List[VectorSimilarityScores]
            A list of tuples where index 0 is a numpy array of the indices of similar vectors and
            index 1 is a numpy array of their cosine similarity scores.
            Tuple i will correspond to the provided comparison_vectors i.
    """
    num_vectors, vectors = __ensure_2d_np_array(comparison_vectors)
    if num_vectors == 0:
        return []
    if len(comparison_embedding.shape) != 2:
        raise ValueError(
            f"Embedding must be 2D Array, provided {len(comparison_embedding.shape)}D"
        )
    num_ignore, ignore_indices_array = __ensure_np_array(ignore_indices)
    # This parallelizes nicely
    similarity_scores = 1 - sklearn.metrics.pairwise_distances(
        vectors, comparison_embedding, metric="cosine"
    )
    relevant_indices = np.flip(np.argsort(similarity_scores), axis=1)

    if cutoff_args is None:
        cutoff_args = {
            "distance_metric": "manhattan",
            "first_elbow": True,
            "max_first_delta": 0.33,
            "below_line_exclusive": True,
        }
    # Need to broadcast this for each if we are multiple vectors at once
    # TODO: Decide whether or not the values should be dropped for finding an elbow
    # If the only thing that is similar is itself then nothing should be returned
    if num_ignore > 0:
        # Need to ensure we ignore things correctly for finding an elbow
        fancy_indices = np.setdiff1d(
            np.arange(comparison_embedding.shape[0]), ignore_indices_array
        )
        elbow_indices = np.apply_along_axis(
            find_cutoff,
            arr=similarity_scores[:, fancy_indices],
            axis=1,
            cutoff_heuristic=cutoff_heuristic,
            **cutoff_args,
        )
    else:
        elbow_indices = np.apply_along_axis(
            find_cutoff,
            arr=similarity_scores,
            axis=1,
            cutoff_heuristic=cutoff_heuristic,
            **cutoff_args,
        )
    # Now I reshape the individual vectors
    # NumPy doesn't support jagged arrays, so now is time to iterate
    result = []
    # An elbow index of -1 indicates we have nothing
    # Otherwise we want the elbow index to be INCLUSIVE
    # An elbow index of values.size should be impossible as by
    # definition it will have the same distance as 0
    elbow_indices += 1
    for row in range(relevant_indices.shape[0]):
        if topn is not None:
            cutoff = min(elbow_indices[row], topn)
        else:
            cutoff = elbow_indices[row]
        if num_ignore > 0:
            item_indices = np.setdiff1d(
                relevant_indices[row], ignore_indices_array, assume_unique=True
            )[:cutoff]
        else:
            item_indices = relevant_indices[row, :cutoff]
        item_scores = similarity_scores[row, item_indices]
        if require_positive and item_scores[item_scores <= 0].size > 0:
            new_cutoff = np.argmax(item_scores <= 0)
            item_indices = item_indices[:new_cutoff]
            item_scores = item_scores[:new_cutoff]
        result.append(VectorSimilarityScores(item_indices, item_scores))
    return result


def find_closest_items_to_average(
    comparison_embedding: NDArray,
    positive: Optional[ArrayLike] = None,
    ignore_positive_indices: Optional[ArrayLike] = None,
    negative: Optional[ArrayLike] = None,
    ignore_negative_indices: Optional[ArrayLike] = None,
    topn: Optional[int] = 100,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find the top-N most similar vectors while also using an elbow-finding heuristic.
    Positive vectors contribute positively towards the similarity, negative vectors negatively.

    This method computes cosine similarity between a simple mean of the projection
    weight vectors of the given vectors and each vector in the model provided embedding.
    It then returns the min(topn, elbow) closest indices.

    Parameters
    ----------
    comparison_embedding: NDarray
        A 2D numpy array of floats.
        Will be compared to vectors and saved if "close enough".
    positive : Optional[ArayLike]
        Zero or more vectors that contribute positively.
    ignore_positive_indices: Optional[ArrayLike]
        The set of indices which represent `positive` values if
        the vectors being examined are from the same array as `comparison_embedding`.
    negative : Optional[ArayLike]
        Zero or more vectors that contribute negatively.
    ignore_negative_indices: Optional[ArrayLike]
        The set of indices which represent `negative` values if
        the vectors being examined are from the same array as `comparison_embedding`.
    topn: Optional[int]
        A maximum number of points to consider similar based on the
        elbow finding heuristic. The number of returned similarity
        scores will be the minimum of the elbow cut-off and topn
        if provided.
    require_positive: bool (Optional, default True)
        If True then only scores which are greater than 0 will be returned.
    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which cutoff heuristic to use.
        See `top2vec.cutoff_heuristics` for more.
    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.find_cutoff` for more information.

    Returns
    -------
    VectorSimilarityScores
        A tuple where index 0 is a numpy array of the indices of similar vectors and
        index 1 is a numpy array of their cosine similarity scores.

    Notes
    -----
    Providing `ignore_*_indices` will prevent any of those being returned
    in the final output.
    This is generally intended for saying "find me words like X which are not exactly X".

    See Also
    --------
    gensim.models.keyedvectors.KeyedVectors.most_similar
    """

    n_positive, positive_array = __ensure_2d_np_array(positive)
    n_negative, negative_array = __ensure_2d_np_array(negative)

    if (n_positive + n_negative) == 0:
        raise ValueError("Must provide positive and/or negative to compute similarity.")

    vector_list = []
    if n_positive != 0:
        vector_list.append(positive_array)
    if n_negative != 0:
        vector_list.append(negative_array * -1)
    # mean_vector = gensim.matutils.unitvec(
    #     np.array(np.vstack(vector_list)).mean(axis=0)
    # ).astype(np.float64)
    # Cosine similarity doesn't care about magnitude
    mean_vector = np.array(np.vstack(vector_list)).mean(axis=0)

    n_positive_indices, positive_indices = __ensure_np_array(ignore_positive_indices)
    n_negative_indices, negative_indices = __ensure_np_array(ignore_negative_indices)
    if n_positive_indices != 0 and n_negative_indices != 0:
        # need to make sure we don't have duplicates
        ignore_indices = np.union1d(positive_indices, negative_indices)
    elif n_positive_indices != 0:
        ignore_indices = positive_indices
    elif n_negative_indices != 0:
        ignore_indices = negative_indices
    else:
        ignore_indices = None

    return find_closest_items(
        mean_vector,
        comparison_embedding=comparison_embedding,
        ignore_indices=ignore_indices,
        topn=topn,
        require_positive=require_positive,
        cutoff_heuristic=cutoff_heuristic,
        cutoff_args=cutoff_args,
    )[0]


def find_similar_in_embedding(
    embedding,
    positive_indices: Optional[ArrayLike] = None,
    negative_indices: Optional[ArrayLike] = None,
    topn: Optional[int] = 100,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
) -> Tuple[NDArray[np.int64], NDArray[np.float64]]:
    """Find the top-N most similar vectors within an embedding while also using
    an elbow-finding heuristic.
    Positive vectors contribute positively towards the similarity, negative vectors negatively.

    Parameters
    ----------
    embedding: NDarray
        A 2D numpy array of floats to find similar vectors within.
    positive_indices: Optional[ArrayLike]
        The set of indices which represent positive values.
    negative_indices: Optional[ArrayLike]
        The set of indices which represent disssimlar values.
    topn: Optional[int]
        A maximum number of points to consider similar based on the
        elbow finding heuristic. The number of returned similarity
        scores will be the minimum of the elbow cut-off and topn
        if provided.
    require_positive: bool (Optional, default True)
        If True then only scores which are greater than 0 will be returned.
    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which cutoff heuristic to use.
        See `top2vec.cutoff_heuristics` for more.
    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.find_cutoff` for more information.

    Returns
    -------
    VectorSimilarityScores
        A tuple where index 0 is a numpy array of the indices of similar vectors and
        index 1 is a numpy array of their cosine similarity scores.

    See Also
    --------
    find_closest_items_to_average
        Allows for commparing to vectors which aren't present in the embedding.
    """
    n_positive_indices, positive_indices = __ensure_np_array(positive_indices)
    n_negative_indices, negative_indices = __ensure_np_array(negative_indices)
    if n_positive_indices > 0:
        positive_vectors = embedding[positive_indices]
    else:
        positive_vectors = None
    if n_negative_indices > 0:
        negative_vectors = embedding[negative_indices]
    else:
        negative_vectors = None
    return find_closest_items_to_average(
        embedding,
        positive=positive_vectors,
        ignore_positive_indices=positive_indices,
        negative=negative_vectors,
        ignore_negative_indices=negative_indices,
        topn=topn,
        require_positive=require_positive,
        cutoff_heuristic=cutoff_heuristic,
        cutoff_args=cutoff_args,
    )


def describe_closest_items(
    vectors: NDArray[np.float64],
    embedding: NDArray[np.float64],
    embedding_vocabulary: ArrayLike,
    topn: int = 100,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
) -> List[Tuple[NDArray, NDArray[np.float64]]]:
    """Finds the most similar embedded vectors for a vector or set of vectors using cosine
    similarity and an elbow finding heuristic.

    Providing the topic vectors as `vectors`, the term embeddings as `embeding`
    and the term vocabulary as `embedding_vocabulary` will result in a description
    of topics in a human readable format. Doing the same with document vectors
    will result in the model's representation of a document x term matrix.

    Parameters
    ----------
    vectors: ArrayLike
        Something which can be interpreted as a 1D or 2D numpy array of
        floats. Will be used as the points to compute distance from.
    embedding: NDarray
        A 2D numpy array of floats.
        Will be compared to vectors and saved if "close enough".
    embedding_vocabulary: ArrayLike
        Something which can be interpreted as a 1D numpy array of
        strings. Index 0 is the human readable description of
        embedding index 0.
    topn: Optional[int] (Optional, default `100`)
        A maximum number of points to consider similar based on the
        elbow finding heuristic. The number of returned similarity
        scores will be the minimum of the elbow cut-off and topn
        if provided.
        Pass `None` to only use the cutoff value.
    require_positive: bool (Optional, default True)
        If True then only scores which are greater than 0 will be returned.
    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which cutoff heuristic to use.
        See `top2vec.cutoff_heuristics` for more.
    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.find_cutoff` for more information.


    Returns
    -------
    List[Tuple[NDArray, NDArray[np.float64]]]
        A list of tuples where index 0 is a numpy array of the similar embedded vectors'
        names (according to vocabulary) and index 1 is a numpy array of their cosine
        similarity scores.
        Tuple i will correspond to the provided vector i.


    Raises
    ------
    ValueError if the vocabulary length is not equal to the 1st dimension of the embedding.

    Notes
    -----
    This will be much more efficient if vocabulary reduction has already been performed.
    """
    vocab_len, vocab_array = __ensure_np_array(embedding_vocabulary)
    if vocab_len != embedding.shape[0]:
        raise ValueError(
            f"Vocabulary size ({vocab_len}) != vocabulary embedding size ({embedding.shape[0]})"
        )
    closest_terms = find_closest_items(
        vectors,
        embedding,
        topn=topn,
        require_positive=require_positive,
        cutoff_heuristic=cutoff_heuristic,
        cutoff_args=cutoff_args,
    )
    results = []
    for indices, scores in closest_terms:
        results.append((vocab_array[indices], scores))
    return results


def generate_similarity_matrix(
    vectors: ArrayLike,
    comparison_embeddings: ArrayLike,
    topn: Optional[int] = 100,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
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
    topn: int (Optional, default `100`)
        A maximum number of points to consider similar to a provided
        vector. Can be used to limit topic sizes.
        Pass `None` to only use the cutoff value.
    require_positive: bool (Optional, default True)
        If True then only scores which are greater than 0 will be returned.
    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which cutoff heuristic to use.
        See `top2vec.cutoff_heuristics` for more.
    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.find_cutoff` for more information.

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
    num_vectors, vector_array = __ensure_np_array(vectors)
    num_embeddings, embeddings_array = __ensure_np_array(comparison_embeddings)
    similarity_values = find_closest_items(
        vector_array,
        embeddings_array,
        topn=topn,
        require_positive=require_positive,
        cutoff_heuristic=cutoff_heuristic,
        cutoff_args=cutoff_args,
    )
    res_matrix = np.zeros((num_vectors, num_embeddings))
    for index, (indices, scores) in enumerate(similarity_values):
        res_matrix[index][indices] = scores
    return res_matrix


def generate_csr_similarity_matrix(
    vectors: ArrayLike,
    comparison_embeddings: ArrayLike,
    topn: Optional[int] = 100,
    require_positive: bool = True,
    cutoff_heuristic: str = ELBOW_HEURISTIC_STR,
    cutoff_args: Optional[Dict] = None,
) -> scipy.sparse.csr_matrix:
    """As with `generate_similarity_matrix`, but a sparse output.

    Translates from a series of vectors and a set of embeddings to compare
    into a CSR matrix. Uses the elbow finding heuristic to determine what is
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
    topn: int
        A maximum number of points to consider similar to a provided
        vector. Can be used to limit topic sizes.
        Pass `None` to only use the cutoff value for size.
    require_positive: bool (Optional, default True)
        If True then only scores which are greater than 0 will be returned.
    cutoff_heuristic: str (Optional, default `'elbow'`)
        Which cutoff heuristic to use.
        See `top2vec.cutoff_heuristics` for more.
    cutoff_args: dict (Optional, default None)
        Pass custom arguments to the cutoff heuristic.
        See `top2vec.cutoff_heurstics.find_cutoff` for more information.

    Returns
    -------
    scipy.sparse.csr_matrix
        A matrix with N rows and M columns.
        N is equal to the number of provided vectors and M is equal
        to the number of provided embeddings.
        The ith row will have values of the cosine similarity for
        all columns [j0, j1, ..., jn] which were deemed to be
        close enough to vector i based on the elbow finding heuristic.

    Notes
    -----
    Assuming that *most* rows will have at least one value,
    therefore we are doing CSR instad of CSC.

    See Also
    --------
    :func:generate_similarity_matrix
    """
    num_vectors, vector_array = __ensure_np_array(vectors)
    num_embeddings, embeddings_array = __ensure_np_array(comparison_embeddings)
    similarity_values = find_closest_items(
        vector_array,
        embeddings_array,
        topn=topn,
        require_positive=require_positive,
        cutoff_heuristic=cutoff_heuristic,
        cutoff_args=cutoff_args,
    )

    res_shape = (num_vectors, num_embeddings)

    res_rows = []
    res_cols = []
    res_values = []

    for index, (col_indices, values) in enumerate(similarity_values):
        # are we empty?
        if len(col_indices) > 0:
            res_rows.append(np.full((len(col_indices)), fill_value=index))
            res_cols.append(col_indices)
            res_values.append(values)

    return scipy.sparse.csc_matrix(
        (
            np.concatenate(res_values),
            (np.concatenate(res_rows), np.concatenate(res_cols)),
        ),
        shape=res_shape,
    )
