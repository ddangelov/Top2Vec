import pytest
import numpy as np
from top2vec.similarity import (
    describe_closest_items,
    find_closest_items,
    find_closest_items_to_average,
    find_similar_in_embedding,
    generate_similarity_matrix,
    generate_csr_similarity_matrix,
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
        return (array_a.round(decimals=5) == array_b.round(decimals=5)).all()
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

    assert find_closest_items(None, test_embedding) == []

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
    with pytest.raises(ValueError):
        find_closest_items([1, 2, 3], np.array([2, 3, 4]))

    # Now test when we ignore some indices
    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
            [4, 2],
            [8, 4],
            [1, -1],
        ]
    )
    # NOTE: This appears to behave poorly when we have values which cross the
    # expected line in a sort of

    res = find_closest_items(test_vectors, test_embedding, ignore_indices=None)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4, 2, 1]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, 1.0, 1.0]), round=True)

    res = find_closest_items(test_vectors, test_embedding, ignore_indices=None, topn=2)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(
        res[0][0],
        np.array(
            [
                5,
                4,
            ]
        ),
    )
    assert compare_numpy_arrays(
        res[0][1],
        np.array(
            [
                1.0,
                1.0,
            ]
        ),
        round=True,
    )

    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
            [4, 2],
            [8, 4],
            [1, -1],
            [1, -1.1],
        ]
    )
    # What about if we ignore everything?
    res = find_closest_items(
        test_vectors, test_embedding, ignore_indices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    )
    for indices, scores in res:
        assert len(indices) == 0
        assert len(scores) == 0
    # Also can't determine closest if given 2 or fewer comparison points
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[
            0,
            1,
            2,
            3,
            4,
            5,
            6,
        ],
    )
    for indices, scores in res:
        assert len(indices) == 0
        assert len(scores) == 0
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[
            0,
            1,
            2,
            3,
            4,
            5,
        ],
    )
    for indices, scores in res:
        assert len(indices) == 0
        assert len(scores) == 0

    # Now real tests
    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[4, 5])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([2, 1]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0]), round=True)

    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[1, 2])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0]), round=True)
    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[1])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4, 2]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, 1.0]), round=True)
    res = find_closest_items(
        test_vectors, test_embedding, ignore_indices=np.array([1, 4])
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 2]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0]), round=True)

    a = test_vectors[1]
    b0 = test_embedding[0]
    b3 = test_embedding[3]
    cosine0 = np.dot(a, b0) / (np.linalg.norm(a) * np.linalg.norm(b0))
    cosine3 = np.dot(a, b3) / (np.linalg.norm(a) * np.linalg.norm(b3))
    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[1, 2])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[1][0], np.array([0]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine0]), round=True)
    res = find_closest_items(
        test_vectors, test_embedding, ignore_indices=[0, 1, 2, 4, 5]
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    # Our indices should be based on the same array we passed in
    assert compare_numpy_arrays(res[1][0], np.array([3]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine3]), round=True)

    # 0 vectors shouldn't be similar to anything
    assert len(res[2][0]) == 0
    assert len(res[2][1]) == 0


def test_find_closest_items_with_averages():
    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
            [4, 2],
            [8, 4],
            [1, -1],
        ]
    )

    with pytest.raises(ValueError):
        find_closest_items_to_average(test_embedding)

    def test_helper(vecs, embedding, ignore_indices, expected_indices, expected_scores):
        pvecs = vecs
        nvecs = np.array(vecs) * -1
        result_tuples = []
        result_tuples.append(
            find_closest_items_to_average(
                embedding, positive=pvecs, ignore_positive_indices=ignore_indices
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                negative=[],
                ignore_positive_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding, positive=pvecs, ignore_negative_indices=ignore_indices
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                ignore_positive_indices=ignore_indices,
                ignore_negative_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=[],
                negative=nvecs,
                ignore_positive_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding, negative=nvecs, ignore_positive_indices=ignore_indices
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding, negative=nvecs, ignore_negative_indices=ignore_indices
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                negative=nvecs,
                ignore_positive_indices=ignore_indices,
                ignore_negative_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                negative=nvecs,
                ignore_positive_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                negative=nvecs,
                ignore_negative_indices=ignore_indices,
            )
        )
        result_tuples.append(
            find_closest_items_to_average(
                embedding,
                positive=pvecs,
                negative=nvecs,
                ignore_positive_indices=ignore_indices,
                ignore_negative_indices=ignore_indices,
            )
        )

        for indices, scores in result_tuples:
            assert len(indices) == len(expected_indices)
            assert len(scores) == len(expected_scores)
            assert len(indices) == len(scores)
            assert compare_numpy_arrays(indices, expected_indices)
            assert compare_numpy_arrays(scores, expected_scores, round=True)

    test_helper(
        [[4, 2]],
        test_embedding,
        [],
        np.array([5, 4, 2, 1]),
        np.array([1.0, 1.0, 1.0, 1.0]),
    )
    test_helper(
        [[4, 2]], test_embedding, [4, 5], np.array([2, 1]), np.array([1.0, 1.0])
    )
    # Make sure duplicates don't cause problems
    test_helper(
        [[4, 2], [4, 2], [4, 2]],
        test_embedding,
        [4, 5],
        np.array([2, 1]),
        np.array([1.0, 1.0]),
    )
    # Easy case: parallel vectors should return identical values
    test_helper(
        [[2, 1], [4, 2]], test_embedding, [4, 5], np.array([2, 1]), np.array([1.0, 1.0])
    )
    test_helper(
        [[2, 1], [4, 2], [6, 3]],
        test_embedding,
        [2, 1],
        np.array([5, 4]),
        np.array([1.0, 1.0]),
    )

    # Slightly different case: providing two orthogonal vectors which average
    # to our value
    test_helper(
        [[2, 0], [0, 1]],
        test_embedding,
        [2, 1],
        np.array([5, 4]),
        np.array([1.0, 1.0]),
    )
    test_helper(
        [[2, 0], [0, 1], [200, 0], [0, 100], [200000, 0], [0, 100000]],
        test_embedding,
        [2, 1],
        np.array([5, 4]),
        np.array([1.0, 1.0]),
    )


def test_find_similar_in_embedding():
    # This is just a wrapper around find_closest_items_to_average
    test_embedding = np.array(
        [
            [0, 1],
            [2, 1],
            [1, 0.5],
            [1, 0],
            [4, 2],
            [8, 4],
            [0, 0],
            [-2, -1],
        ]
    )
    with pytest.raises(ValueError):
        find_similar_in_embedding(None)
    with pytest.raises(TypeError):
        find_similar_in_embedding(None, positive_indices=[1])
    scores, indices = find_similar_in_embedding(test_embedding, positive_indices=[1])
    assert len(scores) == len(indices) == 3
    compare_numpy_arrays(indices, np.array([5, 4, 2]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, 1.0]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[1], topn=2
    )
    assert len(scores) == len(indices) == 2
    compare_numpy_arrays(indices, np.array([5, 4]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0]), round=True)

    scores, indices = find_similar_in_embedding(test_embedding, positive_indices=[4, 5])
    assert len(scores) == len(indices) == 2
    compare_numpy_arrays(indices, np.array([2, 1]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[4, 5], negative_indices=[]
    )
    assert len(scores) == len(indices) == 2
    compare_numpy_arrays(indices, np.array([2, 1]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[], negative_indices=[7]
    )
    assert len(scores) == len(indices) == 4
    compare_numpy_arrays(indices, np.array([5, 4, 2, 1]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, 1.0, 1.0]), round=True)


def test_describe_closest_items():
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
        describe_closest_items(test_vectors, test_embedding, [])
    with pytest.raises(ValueError):
        describe_closest_items(test_vectors, test_embedding[:2], test_vocabulary)
    assert describe_closest_items([], test_embedding, test_vocabulary) == []

    full_run = describe_closest_items(test_vectors, test_embedding, test_vocabulary)
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
    full_run = describe_closest_items(
        test_vectors, test_embedding, test_vocabulary, topn=1
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
    runA = describe_closest_items(test_vectors[0], test_embedding, test_vocabulary)
    runB = describe_closest_items([2, 1], test_embedding, test_vocabulary)

    assert len(runA) == len(runB)
    for idx in range(len(runA)):
        assert len(runA[idx]) == 2
        assert len(runB[idx]) == 2
        assert compare_numpy_arrays(runA[idx][0], runB[idx][0])
        assert compare_numpy_arrays(runA[idx][1], runB[idx][1])

    runB = describe_closest_items(test_vectors[0], test_embedding, test_vocabulary_list)
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


def test_generate_csr_similarity_matrix():
    # This is going to be simple.
    # For each thing I'm going to generate a csr, then ensure it is in fact
    # the same as the numpy array that made it
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
    a = test_topic_vectors_array[1]
    b = test_term_embedding_array[0]
    expected_array = np.array(
        [
            [0, 1, 1, 0],
            [np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    sparse = generate_csr_similarity_matrix(
        test_topic_vectors_array, test_term_embedding_array
    )
    assert compare_numpy_arrays(
        sparse.toarray(),
        expected_array,
        round=True,
    )
    assert not compare_numpy_arrays(sparse.toarray(), np.zeros((3, 4)))

    # Weirder cases
    sparse = generate_csr_similarity_matrix(
        test_topic_vectors_list, test_term_embedding_array
    )
    assert compare_numpy_arrays(sparse.toarray(), expected_array, round=True)
