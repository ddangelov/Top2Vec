import pytest
import numpy as np
import sklearn.metrics
from top2vec.cutoff_heuristics.similarity import (
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


def test_compare_numpy_arrays():
    assert compare_numpy_arrays(None, None)
    assert not compare_numpy_arrays(np.array([1, 2, 3]), None)
    assert compare_numpy_arrays(np.array([]), np.array([]))
    assert compare_numpy_arrays(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert compare_numpy_arrays(
        np.array([1, 2, 3]), np.array([1.0000001, 2.0000001, 3.0000001]), round=True
    )
    assert not compare_numpy_arrays(np.array([1, 2, 3]), np.array([]))
    assert not compare_numpy_arrays(np.array([[1, 2, 3]]), np.array([1, 2, 3]))
    assert not compare_numpy_arrays(np.array([[1, 1], [1, 1]]), np.array([1, 1, 1, 1]))


def get_cosine_sim(vector_a, vector_b):
    if np.linalg.norm(vector_a) == 0 or np.linalg.norm(vector_b) == 0:
        return 0
    return np.dot(vector_a, vector_b) / (
        np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    )


def test_get_cosine_sim():
    # Make sure that 0 doesn't give us issues
    assert get_cosine_sim(np.array([0, 0, 0]), np.array([1, 2, 3])) == 0
    assert get_cosine_sim(np.array([1, 2, 3]), np.array([1, 2, 3])) == 1
    assert get_cosine_sim(np.array([-1, 2]), np.array([2, 1])) == 0


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
    cosine0_3 = get_cosine_sim(test_vectors[0], test_embedding[3])

    assert find_closest_items(None, test_embedding) == []

    res = find_closest_items(test_vectors, test_embedding)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, cosine0_3]), round=True)

    res = find_closest_items(test_vectors_list, test_embedding)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, cosine0_3]), round=True)

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

    res = find_closest_items(test_vectors, test_embedding, ignore_indices=None)
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4, 2, 1, 3]))
    assert compare_numpy_arrays(
        res[0][1], np.array([1.0, 1.0, 1.0, 1.0, cosine0_3]), round=True
    )

    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=None,
        cutoff_args={"max_first_delta": 0},
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(
        res[0][0],
        np.array(
            [
                5,
            ]
        ),
    )
    assert compare_numpy_arrays(
        res[0][1],
        np.array(
            [
                1.0,
            ]
        ),
        round=True,
    )
    assert compare_numpy_arrays(
        res[0].scores,
        np.array(
            [
                1.0,
            ]
        ),
        round=True,
    )
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
    # Elbow finding returns index 0 if given 2 or fewer comparison points
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
        require_positive=True,
    )
    # When we don't require positive
    assert len(res[0][0]) == len(res[0][1]) == 1
    assert len(res[1][0]) == len(res[1][1]) == 0
    assert len(res[2][0]) == len(res[2][1]) == 0

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
        require_positive=False,
    )
    # When we don't require positive
    assert len(res[0][0]) == len(res[0][1]) == 1
    assert len(res[1][0]) == len(res[1][1]) == 1
    assert len(res[2][0]) == len(res[2][1]) == 0
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
        require_positive=True,
    )
    # When we don't require positive
    assert len(res[0][0]) == len(res[0][1]) == 1
    assert len(res[1][0]) == len(res[1][1]) == 0
    assert len(res[2][0]) == len(res[2][1]) == 0
    # Now real tests
    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[4, 5])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([2, 1, 3]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, cosine0_3]), round=True)

    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[1, 2])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4, 3]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, cosine0_3]), round=True)
    res = find_closest_items(test_vectors, test_embedding, ignore_indices=[1])
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 4, 2, 3]))
    assert compare_numpy_arrays(
        res[0][1], np.array([1.0, 1.0, 1.0, cosine0_3]), round=True
    )
    res = find_closest_items(
        test_vectors, test_embedding, ignore_indices=np.array([1, 4])
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[0][0], np.array([5, 2, 3]))
    assert compare_numpy_arrays(res[0][1], np.array([1.0, 1.0, cosine0_3]), round=True)

    cosine1_0 = get_cosine_sim(test_vectors[1], test_embedding[0])
    cosine1_3 = get_cosine_sim(test_vectors[1], test_embedding[3])
    cosine1_6 = get_cosine_sim(test_vectors[1], test_embedding[6])
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[
            1,
            2,
        ],
        require_positive=False,
        cutoff_args={
            "max_first_delta": None,
        },
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    # Below the curve is exclusive
    assert compare_numpy_arrays(res[1][0], np.array([0]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine1_0]), round=True)
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[
            1,
            2,
        ],
        require_positive=True,
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    assert compare_numpy_arrays(res[1][0], np.array([0]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine1_0]), round=True)
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[0, 1, 2, 4, 5],
        require_positive=False,
        cutoff_args={"max_first_delta": None, "below_line_exclusive": False},
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    # Our indices should be based on the same array we passed in
    # Again below curve
    assert compare_numpy_arrays(res[1][0], np.array([3, 6]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine1_3, cosine1_6]), round=True)
    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[0, 1, 2, 4, 5],
        require_positive=False,
        cutoff_args={"max_first_delta": None, "below_line_exclusive": True},
    )
    for indices, scores in res:
        assert len(indices) == len(scores)
    # Our indices should be based on the same array we passed in
    # Again below curve
    assert compare_numpy_arrays(res[1][0], np.array([3]))
    assert compare_numpy_arrays(res[1][1], np.array([cosine1_3]), round=True)

    res = find_closest_items(
        test_vectors,
        test_embedding,
        ignore_indices=[0, 1, 2, 4, 5],
        require_positive=True,
    )
    for indices, scores in res:
        assert len(indices) == len(scores)

    # Everything is negative
    assert compare_numpy_arrays(res[1][0], np.array([]))
    assert compare_numpy_arrays(res[1][1], np.array([]), round=False)

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
        np.array([5, 4, 2, 1, 3]),
        np.array([1.0, 1.0, 1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )
    test_helper(
        [[4, 2]],
        test_embedding,
        [4, 5],
        np.array([2, 1, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )
    # Make sure duplicates don't cause problems
    test_helper(
        [[4, 2], [4, 2], [4, 2]],
        test_embedding,
        [4, 5],
        np.array([2, 1, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )
    # Easy case: parallel vectors should return identical values
    test_helper(
        [[2, 1], [4, 2]],
        test_embedding,
        [4, 5],
        np.array([2, 1, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )
    test_helper(
        [[2, 1], [4, 2], [6, 3]],
        test_embedding,
        [2, 1],
        np.array([5, 4, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )

    # Slightly different case: providing two orthogonal vectors which average
    # to our value
    test_helper(
        [[2, 0], [0, 1]],
        test_embedding,
        [2, 1],
        np.array([5, 4, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
    )
    test_helper(
        [[2, 0], [0, 1], [200, 0], [0, 100], [200000, 0], [0, 100000]],
        test_embedding,
        [2, 1],
        np.array([5, 4, 3]),
        np.array([1.0, 1.0, get_cosine_sim([4, 2], test_embedding[3])]),
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

    cosines = 1 - sklearn.metrics.pairwise_distances(
        test_embedding, test_embedding, metric="cosine"
    )

    with pytest.raises(ValueError):
        find_similar_in_embedding(None)
    with pytest.raises(TypeError):
        find_similar_in_embedding(None, positive_indices=[1])
    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[1], require_positive=True
    )
    assert len(scores) == len(indices) == 4
    compare_numpy_arrays(indices, np.array([5, 4, 2, 0]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, 1.0, cosines[1][0]]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[1], require_positive=False
    )
    assert len(scores) == len(indices) == 4
    compare_numpy_arrays(indices, np.array([5, 4, 2, 0]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, 1.0, cosines[1][0]]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[1], topn=2
    )
    assert len(scores) == len(indices) == 2
    compare_numpy_arrays(
        indices,
        np.array(
            [
                5,
                4,
            ]
        ),
    )
    compare_numpy_arrays(scores, np.array([1.0, 1.0]), round=True)

    scores, indices = find_similar_in_embedding(test_embedding, positive_indices=[4, 5])
    assert len(scores) == len(indices) == 3
    compare_numpy_arrays(indices, np.array([2, 1, 0]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, cosines[1][0]]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[4, 5], negative_indices=[]
    )
    assert len(scores) == len(indices) == 3
    compare_numpy_arrays(indices, np.array([2, 1, 0]))
    compare_numpy_arrays(scores, np.array([1.0, 1.0, cosines[1][0]]), round=True)

    scores, indices = find_similar_in_embedding(
        test_embedding, positive_indices=[], negative_indices=[7]
    )
    assert len(scores) == len(indices) == 5
    compare_numpy_arrays(indices, np.array([5, 4, 2, 1, 0]))
    compare_numpy_arrays(
        scores, np.array([1.0, 1.0, 1.0, 1.0, cosines[1][0]]), round=True
    )

    scores0, indices0 = find_similar_in_embedding(
        test_embedding, positive_indices=[1], topn=2
    )
    scores1, indices1 = find_closest_items(
        test_embedding[1], test_embedding, topn=2, ignore_indices=[1]
    )[0]

    # We should get an identical value when running find_similar
    # and find_embedding
    assert compare_numpy_arrays(scores0, scores1, round=True)
    assert compare_numpy_arrays(indices0, indices1, round=True)


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
    cosines = 1 - sklearn.metrics.pairwise_distances(
        test_vectors, test_embedding, metric="cosine"
    )

    with pytest.raises(ValueError):
        describe_closest_items(test_vectors, test_embedding, [])
    with pytest.raises(ValueError):
        describe_closest_items(test_vectors, test_embedding[:2], test_vocabulary)
    assert describe_closest_items([], test_embedding, test_vocabulary) == []

    full_run = describe_closest_items(test_vectors, test_embedding, test_vocabulary)
    assert len(full_run) == test_vectors.shape[0]
    # We are inclusive, so two with 1 and the other close item
    assert full_run[0][0].shape == (3,)
    assert full_run[0][1].shape == (3,)
    assert compare_numpy_arrays(
        full_run[0][1], np.array([1, 1, cosines[0][3]]), round=True
    )
    assert compare_numpy_arrays(full_run[0][0], np.array(["kitten", "cat", "car"]))

    assert full_run[1][0].shape == (1,)
    assert full_run[1][1].shape == (1,)
    # cosine similarity
    assert compare_numpy_arrays(
        full_run[1][1],
        np.array([cosines[1][0]]),
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
        np.array([cosines[1][0]]),
        round=True,
    )
    assert compare_numpy_arrays(full_run[1][0], np.array(["hat"]))
    # All 0s doesn't really work with anything
    assert full_run[2][0].shape == (0,)
    assert full_run[2][1].shape == (0,)

    # Testing with no max first delta or requiring positive
    full_run = describe_closest_items(
        test_vectors,
        test_embedding,
        test_vocabulary,
        require_positive=False,
        cutoff_args={"below_line_exclusive": False, "max_first_delta": None},
    )
    assert len(full_run) == test_vectors.shape[0]
    # Two items with cosine similarity of 1
    assert full_run[0][0].shape == (3,)
    assert full_run[0][1].shape == (3,)
    assert compare_numpy_arrays(
        full_run[0][1], np.array([1, 1, cosines[0][3]]), round=True
    )
    assert compare_numpy_arrays(full_run[0][0], np.array(["kitten", "cat", "car"]))
    # Just one item, two are orthogonal and one is negative
    assert full_run[1][0].shape == (2,)
    assert full_run[1][1].shape == (2,)
    # cosine similarity
    assert compare_numpy_arrays(
        full_run[1][1],
        np.array([cosines[1][0], 0]),
        round=True,
    )
    assert compare_numpy_arrays(full_run[1][0], np.array(["hat", "kitten"]))
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
    cosine0_3 = get_cosine_sim(
        test_topic_vectors_array[0], test_term_embedding_array[3]
    )
    cosine1_0 = get_cosine_sim(
        test_topic_vectors_array[1], test_term_embedding_array[0]
    )
    # This is negative so the next closest is actually 0 similarity in this case
    expected_array = np.array(
        [
            [0, 1, 1, cosine0_3],
            [cosine1_0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    # The happy case
    res_matrix = generate_similarity_matrix(
        test_topic_vectors_array, test_term_embedding_array, require_positive=False
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
    # Ensure require_positive actually does what we think it does
    rand_vecs = np.random.rand(12, 2)
    rand_embeddings = np.random.rand(128, 2)
    res_matrix = generate_similarity_matrix(
        rand_vecs, rand_embeddings, require_positive=True
    )
    # We WILL have zero values, but no negative values
    assert res_matrix[res_matrix < 0].size == 0


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
    cosine0_3 = get_cosine_sim(
        test_topic_vectors_array[0], test_term_embedding_array[3]
    )
    cosine1_0 = get_cosine_sim(
        test_topic_vectors_array[1], test_term_embedding_array[0]
    )
    # This is negative so the next closest is actually 0 similarity in this case
    expected_array = np.array(
        [
            [0, 1, 1, cosine0_3],
            [cosine1_0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    sparse = generate_csr_similarity_matrix(
        test_topic_vectors_array, test_term_embedding_array, require_positive=False
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
