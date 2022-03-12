import numpy as np
from typing import Optional, Union, NamedTuple
from numpy.typing import NDArray


class LineDistances(NamedTuple):
    """Represents multiple data points about distance from a line.

    Attributes
    ----------
    distances: NDArray[np.float64]
        The absolute value of the distance between value[index i]
        and provided_line[index i].
    y_deltas: NDArray[np.float64]
        The distance between value[index i] and provided_line[index i].
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


class SimilarVectorIndices(NamedTuple):
    """Lists of the most similar to least similar vectors and their
    corresponding similarity scores.

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


class SimilarItems(NamedTuple):
    """Lists of the most similar to least similar items and their
    corresponding similarity scores.

    Attributes
    ----------
    items: NdArray
        An array of items (usually strings) sorted from most to least similar.
    scores: NdArray[np.float64]
        Index 0 is the similarity score of `indices[0]` and the
        original vector.
    """

    indices: NDArray
    scores: NDArray[np.float64]


DocumentId = Union[np.str_, np.int_]


class RetrievedDocuments(NamedTuple):
    """Represents a series of documents that have been returned by a query.

    Attributes
    ----------
    documents: Optional[NDArray[np.str_]]
        The raw documents as strings.
        Will be `None` if the documents are not contained by the
        Top2Vec model or if the search did not request document
        content.

    scores: NDArray[np.float64]
        Cosine similarity scores to the provided query in
        descending order.

    doc_ids: NDArray[DocumentId]
        Index i is the document id with score `scores[i]`
        and raw content `documents[i]`.
    """

    documents: Optional[NDArray[np.str_]]
    scores: NDArray[np.float64]
    doc_ids: NDArray[DocumentId]
