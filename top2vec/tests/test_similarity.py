import pytest
import numpy as np
from top2vec.similarity import (
    describe_topics,
    find_closest_items,
    generate_similarity_matrix,
)


def compare_numpy_arrays(array_a, array_b, round=False):
    if array_a is None and array_b is None:
        return True
    elif array_a is None or array_b is None:
        return False
    # What about our size?
    if array_a.size == 0 and array_b.size == 0:
        return True
    elif array_a.size == 0 or array_b.size == 0:
        return False
    if array_a.shape != array_b.shape:
        return False
    # Thanks a bunch, floating point numbers
    if round:
        return (abs(array_a - array_b) < 0.00000001).all()
    else:
        return (array_a == array_b).all()


def test_find_closest_items():
    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
        ]
    )
    test_vectors_list = [
        [2, 1],
        [-1, 2],
        [0, 0],
    ]
    test_vectors = np.array(test_vectors_list)

    res = find_closest_items(test_vectors, test_embedding)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0]), round=True)

    res = find_closest_items(test_vectors_list, test_embedding)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0]), round=True)

    # We should have problems if we give items that aren't in the same space
    with pytest.raises(ValueError):
        find_closest_items([1, 2, 3], test_embedding)


def test_describe_topics():
    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
        ]
    )
    test_vocabulary_list = ["hat", "cat", "kitten", "car"]
    test_vocabulary = np.array(test_vocabulary_list)

    test_vectors = np.array(
        [
            [2, 1],
            [-1, 2],
            [0, 0],
        ]
    )
    a = test_vectors[1]
    b = test_embedding[0]

    with pytest.raises(ValueError):
        describe_topics(test_vectors, test_embedding, [])
    with pytest.raises(ValueError):
        describe_topics(test_vectors, test_embedding[:2], test_vocabulary)
    with pytest.raises(ValueError):
        describe_topics([], test_embedding, test_vocabulary)

    full_run = describe_topics(test_vectors, test_embedding, test_vocabulary)
    assert len(full_run) == test_vectors.shape[0]
    # Two items with cosine similarity of 1
    assert full_run[0][0].shape == (2,)
    assert full_run[0][1].shape == (2,)
    assert compare_numpy_arrays(full_run[0][1], np.array([1, 1]), round=True)
    assert compare_numpy_arrays(full_run[0][0], np.array(["kitten", "cat"]))
    # Just one item, two are orthogonal and one is negative
    assert full_run[1][0].shape == (1,)
    assert full_run[1][1].shape == (1,)
    # cosine similarity
    assert compare_numpy_arrays(
        full_run[1][1],
        np.array([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))]),
        round=True,
    )
    assert compare_numpy_arrays(full_run[1][0], np.array(["hat"]))
    # All 0s doesn't really work with anything
    assert full_run[2][0].shape == (0,)
    assert full_run[2][1].shape == (0,)

    # But what if we give it a max value?
    full_run = describe_topics(
        test_vectors, test_embedding, test_vocabulary, maxTerms=1
    )
    assert len(full_run) == test_vectors.shape[0]
    # Two items with cosine similarity of 1
    assert full_run[0][0].shape == (1,)
    assert full_run[0][1].shape == (1,)
    assert compare_numpy_arrays(full_run[0][1], np.array([1]), round=True)
    assert compare_numpy_arrays(full_run[0][0], np.array(["kitten"]))
    # Just one item, two are orthogonal and one is negative
    assert full_run[1][0].shape == (1,)
    assert full_run[1][1].shape == (1,)
    # cosine similarity
    assert compare_numpy_arrays(
        full_run[1][1],
        np.array([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))]),
        round=True,
    )
    assert compare_numpy_arrays(full_run[1][0], np.array(["hat"]))
    # All 0s doesn't really work with anything
    assert full_run[2][0].shape == (0,)
    assert full_run[2][1].shape == (0,)

    # Should be identical
    runA = describe_topics(test_vectors[0], test_embedding, test_vocabulary)
    runB = describe_topics([2, 1], test_embedding, test_vocabulary)

    assert len(runA) == len(runB)
    for idx in range(len(runA)):
        assert len(runA[idx]) == 2
        assert len(runB[idx]) == 2
        assert compare_numpy_arrays(runA[idx][0], runB[idx][0])
        assert compare_numpy_arrays(runA[idx][1], runB[idx][1])

    runB = describe_topics(test_vectors[0], test_embedding, test_vocabulary_list)
    assert len(runA) == len(runB)
    for idx in range(len(runA)):
        assert len(runA[idx]) == 2
        assert len(runB[idx]) == 2
        assert compare_numpy_arrays(runA[idx][0], runB[idx][0])
        assert compare_numpy_arrays(runA[idx][1], runB[idx][1])


def test_generate_similarity_matrix():
    test_term_embedding_list = [
        [0, 1],
        [2, 1],
        [1, 0.5],
        [1, 0],
    ]
    test_term_embedding_array = np.array(test_term_embedding_list)
    test_topic_vectors_list = [
        [2, 1],
        [-1, 2],
        [0, 0],
    ]
    test_topic_vectors_array = np.array(test_topic_vectors_list)
    # The happy case
    res_matrix = generate_similarity_matrix(
        test_topic_vectors_array, test_term_embedding_array
    )
    a = test_topic_vectors_array[1]
    b = test_term_embedding_array[0]
    expected_array = np.array(
        [
            [0, 1, 1, 0],
            [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    assert compare_numpy_arrays(
        res_matrix,
        expected_array,
        round=True,
    )
    assert not compare_numpy_arrays(res_matrix, np.zeros((3, 4)))

    # Weirder cases
    res_matrix = generate_similarity_matrix(
        test_topic_vectors_list, test_term_embedding_array
    )
    assert compare_numpy_arrays(
        res_matrix,
        expected_array,
        round=True,
    )
