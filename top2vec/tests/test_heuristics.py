import pytest
from top2vec.Top2Vec import Top2Vec, RetrievedDocuments
from top2vec.cutoff_heuristics.cutoff_heuristics import ELBOW_HEURISTIC_STR
from top2vec.cutoff_heuristics.similarity import (
    describe_closest_items,
    find_closest_items,
    find_similar_in_embedding,
    generate_similarity_matrix,
    generate_csr_similarity_matrix,
)
import gensim
from sklearn.datasets import fetch_20newsgroups
import numpy as np

N_DECIMALS = 4


def compare_numpy_arrays(
    array_a, array_b, round=False, print_why=True, print_verbose=True
):
    if array_a is None and array_b is None:
        return True
    elif array_a is None or array_b is None:
        if print_why:
            print("One is None")
        return False
    # What about our size?
    if array_a.size == 0 and array_b.size == 0:
        return True
    elif array_a.size == 0 or array_b.size == 0:
        if print_why:
            print("Unequal size")
        return False
    if array_a.shape != array_b.shape:
        if print_why:
            print("Unequal shape")
        return False
    # Thanks a bunch, floating point numbers
    if round:
        # Have some items which are .0001 off
        diff = np.abs(
            array_a.round(decimals=N_DECIMALS) - array_b.round(decimals=N_DECIMALS)
        )
        if diff[diff > 0.001].any():
            if print_why:
                print("Arrays more different than 0.001")
                if print_verbose:
                    print(diff)
            return False
        else:
            return True
    else:
        if (array_a == array_b).all():
            return True
        else:
            if print_why:
                diff = np.argwhere(array_a != array_b)
                print("Arrays have differences")
                if print_verbose:
                    print(diff)
                    print(array_a[diff])
                    print(array_b[diff])
            return False


def test_compare_numpy_arrays():
    assert compare_numpy_arrays(None, None)
    assert not compare_numpy_arrays(np.array([1, 2, 3]), None)
    assert compare_numpy_arrays(np.array([]), np.array([]))
    assert compare_numpy_arrays(np.array([1, 2, 3]), np.array([1, 2, 3]))
    assert not compare_numpy_arrays(np.array([1, 2, 3]), np.array([1, 2, 3, 4]))
    assert not compare_numpy_arrays(np.array([1, 2, 3]), np.array([1, 2, 4]))
    assert compare_numpy_arrays(
        np.array([1, 2, 3]), np.array([1.0000001, 2.0000001, 3.0000001]), round=True
    )
    assert not compare_numpy_arrays(
        np.array([1, 2, 3]), np.array([1.0000001, 2.0000001, 4.0000001]), round=True
    )
    assert not compare_numpy_arrays(np.array([1, 2, 3]), np.array([]))
    assert not compare_numpy_arrays(np.array([[1, 2, 3]]), np.array([1, 2, 3]))
    assert not compare_numpy_arrays(np.array([[1, 1], [1, 1]]), np.array([1, 1, 1, 1]))


# get 20 newsgroups data
newsgroups_train = fetch_20newsgroups(
    subset="all", remove=("headers", "footers", "quotes")
)
# newsgroups_documents = newsgroups_train.data[0:2000]
newsgroups_documents = newsgroups_train.data

# train top2vec model with doc_ids provided

doc_ids = [str(num) for num in range(0, len(newsgroups_documents))]
top2vec_docids_cutoff = Top2Vec(
    documents=newsgroups_documents,
    document_ids=doc_ids,
    speed="fast-learn",
    workers=8,
    umap_args={"random_state": 1337},
    use_cutoff_heuristics=True,
)
# test USE with model embedding
# This only gives us 2 topics when given 2000 documents, which isn't great
top2vec_use_model_embedding_cutoff = Top2Vec(
    documents=newsgroups_documents,
    embedding_model="universal-sentence-encoder",
    use_embedding_model_tokenizer=True,
    umap_args={"random_state": 1337},
    use_cutoff_heuristics=True,
)
# test USE-multilang
top2vec_use_multilang_cutoff = Top2Vec(
    documents=newsgroups_documents,
    embedding_model="universal-sentence-encoder-multilingual",
    umap_args={"random_state": 1337},
    use_cutoff_heuristics=True,
)
top2vec_use_cutoff = Top2Vec(
    documents=newsgroups_documents,
    embedding_model="universal-sentence-encoder",
    umap_args={"random_state": 1337},
    use_cutoff_heuristics=True,
)
top2vec_use = Top2Vec(
    documents=newsgroups_documents,
    embedding_model="universal-sentence-encoder",
    umap_args={"random_state": 1337},
    use_cutoff_heuristics=False,
)

models = [
    top2vec_docids_cutoff,
    top2vec_use_model_embedding_cutoff,
    top2vec_use_multilang_cutoff,
    top2vec_use_cutoff,
    top2vec_use,
]


@pytest.mark.parametrize("top2vec_model", models)
def test_topic_descriptions(top2vec_model: Top2Vec):
    # Is topn respected?
    topn = 100
    topic_descriptions = describe_closest_items(
        top2vec_model.topic_vectors,
        top2vec_model.word_vectors,
        top2vec_model.vocab,
        topn=topn,
    )
    topic_lens = [len(words) for (words, scores) in topic_descriptions]
    for topic_len in topic_lens:
        assert topic_len <= topn

    topn = 1000
    topic_descriptions = describe_closest_items(
        top2vec_model.topic_vectors,
        top2vec_model.word_vectors,
        top2vec_model.vocab,
        topn=topn,
    )
    topic_lens = [len(words) for (words, scores) in topic_descriptions]
    for topic_len in topic_lens:
        assert topic_len <= topn

    for topic_num in range(len(topic_descriptions)):
        words, scores = top2vec_model.search_words_by_vector(
            top2vec_model.topic_vectors[topic_num], topn, use_index=False
        )
        expected_words, expected_scores = topic_descriptions[topic_num]
        # The ordering can get slightly off, especially as items get rounded
        if top2vec_model.use_cutoff_heuristics:
            assert scores.size <= topn
            assert set(expected_words) == set(words)
            assert compare_numpy_arrays(expected_scores, scores, round=True)
        else:
            assert scores.size == topn


@pytest.mark.parametrize("top2vec_model", models)
def test_document_descriptions(top2vec_model: Top2Vec):
    # Our average contained will be much lower with a recursive elbow as opposed
    # to traditional elbow, so change the cutoff_args
    old_cutoff_args = top2vec_model.cutoff_args
    top2vec_model.cutoff_args = {"cutoff_heuristic": ELBOW_HEURISTIC_STR}
    # Make sure we don't run out of memory
    maxDocs = 50
    topn = 1000
    document_descriptions = describe_closest_items(
        top2vec_model.document_vectors[:maxDocs],
        top2vec_model.word_vectors,
        top2vec_model.vocab,
        topn=topn,
        cutoff_args={"cutoff_heuristic": ELBOW_HEURISTIC_STR},
    )
    doc_lens = [len(words) for (words, scores) in document_descriptions]
    for doc_len in doc_lens:
        assert doc_len <= topn
    assert len(doc_lens) == min(len(top2vec_model.document_vectors), maxDocs)
    percent_contained_per_doc = []
    for document_index in range(len(document_descriptions)):
        tokenized_doc = gensim.utils.simple_tokenize(
            newsgroups_documents[document_index]
        )
        lower_tokens = set([word.lower() for word in tokenized_doc])
        contained = [
            token
            for token in lower_tokens
            if token in document_descriptions[document_index][0]
        ]
        # So this is an interesting thing.
        # Not all documents will contain their closest word vectors
        percent_contained = len(contained) / len(lower_tokens)
        percent_contained_per_doc.append(percent_contained)
    average_percent_contained_per_doc = sum(percent_contained_per_doc) / len(
        percent_contained_per_doc
    )
    print("Average percent contained per document: ", average_percent_contained_per_doc)
    assert average_percent_contained_per_doc > 0.125

    for document_vector_index in range(maxDocs):
        expected_words, expected_scores = document_descriptions[document_vector_index]
        words, scores = top2vec_model.search_words_by_vector_heuristic(
            top2vec_model.document_vectors[document_vector_index], topn
        )
        assert set(expected_words) == set(words)
        assert compare_numpy_arrays(expected_scores, scores, round=True)

        if top2vec_model.use_cutoff_heuristics:
            words, scores = top2vec_model.search_words_by_vector(
                top2vec_model.document_vectors[document_vector_index], topn
            )
            assert set(expected_words) == set(words)
            assert compare_numpy_arrays(expected_scores, scores, round=True)
    top2vec_model.cutoff_args = old_cutoff_args


@pytest.mark.parametrize("top2vec_model", models)
def test_document_topic_composition(top2vec_model: Top2Vec):
    # Theory is that MOST documents should be composed of a single topic
    # However in this case our USE has almost 100 topics despite there
    # being only 20 newsgroups. Therefore there is some duplication
    topn = 100
    document_topics = find_closest_items(
        top2vec_model.document_vectors,
        top2vec_model.topic_vectors,
        topn=topn,
        cutoff_args=top2vec_model.cutoff_args,
    )
    num_topics_per_doc = [len(scores) for indices, scores in document_topics]
    for num_topics in num_topics_per_doc:
        assert num_topics <= topn

    # Now let's see if the doc x topic matrix looks good
    docTopicMatrix = generate_similarity_matrix(
        top2vec_model.document_vectors,
        top2vec_model.topic_vectors,
        topn=topn,
        cutoff_args=top2vec_model.cutoff_args,
    )
    numZeroes = np.count_nonzero(docTopicMatrix == 0)
    sparsity = numZeroes / (docTopicMatrix.size)
    # Our sparsity should be high
    # It looks like we've got some weirdness where a few of the items
    # have a TON of topics
    assert sparsity > 0.8

    sparse_matrix = generate_csr_similarity_matrix(
        top2vec_model.document_vectors,
        top2vec_model.topic_vectors,
        topn=topn,
        cutoff_args=top2vec_model.cutoff_args,
    )
    assert sparse_matrix.size == np.count_nonzero(docTopicMatrix)

    n_topics = min(topn, len(top2vec_model.topic_vectors) - 1)
    print(n_topics)
    for document_num, (expected_topic_nums, expected_topic_scores) in enumerate(
        document_topics[:50]
    ):
        doc_vector = top2vec_model.document_vectors[document_num]
        assert doc_vector.size == top2vec_model.document_vectors.shape[1]
        top2vec_model._validate_vector(doc_vector)
        (
            topics_words,
            word_scores,
            topic_scores,
            topic_nums,
        ) = top2vec_model.search_topics_by_vector(doc_vector, num_topics=n_topics)
        # tstA = np.round(topic_scores, decimals=N_DECIMALS)
        # tstB = np.round(expected_topic_scores, decimals=N_DECIMALS)
        if top2vec_model.use_cutoff_heuristics:
            assert topic_nums.size <= n_topics
            assert compare_numpy_arrays(
                topic_scores,
                expected_topic_scores,
                round=True,
            )
            assert compare_numpy_arrays(topic_nums, expected_topic_nums)
        else:
            assert topic_nums.size == n_topics


def test_USE_topic_descriptions():
    # This was built assuming an elbow heuristic
    old_cutoff_args = top2vec_use_model_embedding_cutoff.cutoff_args
    top2vec_use_model_embedding_cutoff.cutoff_args = {
        "cutoff_heuristic": ELBOW_HEURISTIC_STR
    }

    topn = 1000
    topic_descriptions = describe_closest_items(
        top2vec_use_model_embedding_cutoff.topic_vectors,
        top2vec_use_model_embedding_cutoff.word_vectors,
        top2vec_use_model_embedding_cutoff.vocab,
        topn=topn,
        cutoff_args=top2vec_use_model_embedding_cutoff.cutoff_args,
    )
    topic_lens = [len(words) for (words, scores) in topic_descriptions]

    # Let's find our space topic
    space_topic = None
    space_topic_num = None

    for topic_len in topic_lens:
        assert topic_len <= topn

    for topic_num, (words, scores) in enumerate(topic_descriptions):
        if "spacecraft" in words[:10]:
            space_topic = words
            space_topic_num = topic_num
            break
    assert space_topic is not None
    assert "jpl" in space_topic[:10]
    assert "orbiter" in space_topic[:10]
    assert "satellites" in space_topic[:10]
    assert "astronaut" in space_topic[:10]
    assert len(words) > 100

    (
        topics_words,
        word_scores,
        topic_scores,
        topic_nums,
    ) = top2vec_use_model_embedding_cutoff.search_topics(
        ["spacecraft", "jpl"], num_topics=80
    )
    assert space_topic_num in topic_nums
    # Not everything should be similar
    assert topic_nums.size < top2vec_use_model_embedding_cutoff.get_num_topics()

    (
        topics_words,
        word_scores,
        topic_scores,
        topic_nums,
    ) = top2vec_use_model_embedding_cutoff.query_topics("spacecraft jpl", num_topics=80)
    assert space_topic_num in topic_nums
    assert topic_nums.size < top2vec_use_model_embedding_cutoff.get_num_topics()

    top2vec_use_model_embedding_cutoff.cutoff_args = old_cutoff_args


def document_return_helper(
    top2vec_model: Top2Vec, search_documents_tuple: RetrievedDocuments, get_docs=True
):
    assert len(search_documents_tuple) == 3
    if get_docs:
        assert search_documents_tuple.documents is not None
    if search_documents_tuple.documents is not None:
        for documents_array_index, top2vec_document_id in enumerate(
            search_documents_tuple.doc_ids
        ):
            top2vec_document_index = top2vec_model._get_document_indexes(
                [top2vec_document_id]
            )[0]
            assert (
                top2vec_model.documents[top2vec_document_index]
                == search_documents_tuple.documents[documents_array_index]
            )
    return search_documents_tuple


@pytest.mark.parametrize("top2vec_model", models)
def test_search_document_by_topic_heuristics(top2vec_model: Top2Vec):
    # As of now there is a different return type on the various search_documents
    # functions. It may be best to change this to be a single named tuple
    # which has None for documents if the function isn't asked to
    # return them

    if top2vec_model.hierarchy is None:
        # perform reduction
        new_number_topics = int(len(top2vec_model.topic_vectors) / 2)
        top2vec_model.hierarchical_topic_reduction(new_number_topics)

    for reduced_topics in [False, True]:
        if reduced_topics:
            t_vectors = top2vec_model.topic_vectors_reduced
        else:
            t_vectors = top2vec_model.topic_vectors
        # With and without returning the original documents
        for return_docs in [True, False]:
            num_docs = 50
            get_docs = return_docs and top2vec_model.documents is not None

            for topic_num in range(len(t_vectors)):
                # The base case: search_documents_by_vector
                if top2vec_model.use_cutoff_heuristics:
                    expected_res = top2vec_model.search_documents_by_vector(
                        t_vectors[topic_num],
                        num_docs,
                        return_documents=return_docs,
                        use_index=False,
                    )
                else:
                    expected_res = top2vec_model.search_documents_by_vector_heuristic(
                        t_vectors[topic_num],
                        num_docs,
                        return_documents=return_docs,
                    )
                (
                    expected_docs,
                    expected_scores,
                    expected_indices,
                ) = document_return_helper(top2vec_model, expected_res, get_docs)
                # Annoyingly the structure is different for this
                assert len(expected_scores) <= num_docs
                assert len(expected_indices) == len(expected_scores)

                # search_documents_by_topic
                if top2vec_model.use_cutoff_heuristics:
                    other_res = top2vec_model.search_documents_by_topic(
                        topic_num,
                        num_docs,
                        return_documents=return_docs,
                        reduced=reduced_topics,
                    )
                else:
                    other_res = top2vec_model.search_documents_by_topic_heuristic(
                        topic_num,
                        num_docs,
                        return_documents=return_docs,
                        reduced=reduced_topics,
                    )
                other_docs, other_scores, other_indices = document_return_helper(
                    top2vec_model, other_res, get_docs
                )
                assert len(other_scores) <= num_docs
                assert len(other_indices) == len(other_scores)

                print(reduced_topics)
                print(topic_num)

                assert compare_numpy_arrays(expected_indices, other_indices)
                assert compare_numpy_arrays(expected_scores, other_scores, round=True)
                assert compare_numpy_arrays(expected_docs, other_docs)
                # We've verified that we are internally consistent


@pytest.mark.parametrize("top2vec_model", models)
def test_search_document_by_keywords_heuristics(top2vec_model):
    # With and without returning the original documents
    for return_docs in [True, False]:
        num_docs = 50
        get_docs = return_docs and top2vec_model.documents is not None

        # For the first bit we aren't going to search with combos
        for word_index in range(min(len(top2vec_model.vocab) - 1, 50)):
            word = top2vec_model.vocab[word_index]
            word_vectors = top2vec_model._words2word_vectors([word])

            # We want a truly identical vector, so we will do everything
            # BUT l2 normalize it as that is the first thing that happens
            # in the search_by_vector function
            combined_vector = np.zeros(word_vectors.shape[1], dtype=np.float64)
            for word_vector in word_vectors:
                combined_vector += word_vector
            combined_vector /= 1
            # Even with running the exact same process we get different values
            # assert compare_numpy_arrays(
            #    top2vec_model._l2_normalize(combined_vector),
            #    top2vec_model._l2_normalize(word_vectors[0]),
            #    print_verbose=True
            # )

            # The base case: search_documents_by_vector
            if top2vec_model.use_cutoff_heuristics:
                vector_res = top2vec_model.search_documents_by_vector(
                    combined_vector,
                    num_docs,
                    return_documents=return_docs,
                    use_index=False,
                )
            else:
                vector_res = top2vec_model.search_documents_by_vector_heuristic(
                    combined_vector,
                    num_docs,
                    return_documents=return_docs,
                )
            vector_docs, vector_scores, vector_indices = document_return_helper(
                top2vec_model, vector_res, get_docs
            )

            assert len(vector_scores) <= num_docs
            assert len(vector_indices) == len(vector_scores)

            if top2vec_model.use_cutoff_heuristics:
                keyword_res = top2vec_model.search_documents_by_keywords(
                    [word],
                    num_docs,
                    return_documents=return_docs,
                    use_index=False,
                )
            else:
                keyword_res = top2vec_model.search_documents_by_keywords_heuristic(
                    [word],
                    num_docs,
                    return_documents=return_docs,
                )
            keyword_docs, keyword_scores, keyword_indices = document_return_helper(
                top2vec_model, keyword_res, get_docs
            )

            assert len(keyword_scores) <= num_docs
            assert len(keyword_indices) == len(keyword_scores)

            assert compare_numpy_arrays(vector_scores, keyword_scores, round=True)
            # Because our vectors are slightly different we get different values
            # here. Therefore I am going to convert to a set and check that
            # the difference is small enough
            if len(vector_indices) != 0 or len(keyword_indices) != 0:
                vector_indices_set = set(vector_indices)
                keyword_indices_set = set(keyword_indices)
                set_differences = vector_indices_set.symmetric_difference(
                    keyword_indices_set
                )
                # we are getting up to 50 documents back, so we want less than 7.5 problems
                assert (
                    len(set_differences)
                    / len(vector_indices_set.union(keyword_indices_set))
                ) <= 0.15
                # assert compare_numpy_arrays(vector_indices, keyword_indices)
            assert compare_numpy_arrays(vector_docs, keyword_docs)

            # We have major problems if our model returns the same thing for positive and negative
            if top2vec_model.use_cutoff_heuristics:
                neg_keyword_res = top2vec_model.search_documents_by_keywords(
                    keywords=[],
                    num_docs=num_docs,
                    keywords_neg=[word],
                    return_documents=return_docs,
                    use_index=False,
                )
            else:
                neg_keyword_res = top2vec_model.search_documents_by_keywords_heuristic(
                    keywords=[],
                    num_docs=num_docs,
                    keywords_neg=[word],
                    return_documents=return_docs,
                )
            (
                neg_keyword_docs,
                neg_keyword_scores,
                neg_keyword_indices,
            ) = document_return_helper(top2vec_model, neg_keyword_res, get_docs)

            assert len(neg_keyword_scores) <= num_docs
            assert len(neg_keyword_indices) == len(neg_keyword_scores)
            assert not compare_numpy_arrays(
                neg_keyword_indices, keyword_indices, print_why=False
            )
            assert not compare_numpy_arrays(
                neg_keyword_scores, keyword_scores, round=True, print_why=False
            )
            if neg_keyword_docs is not None or keyword_docs is not None:
                assert not compare_numpy_arrays(
                    neg_keyword_docs, keyword_docs, print_why=False
                )
    # TODO: Decide if we want to compare what we get from searching by averages as well as individual vectors


@pytest.mark.parametrize("top2vec_model", models)
def test_search_document_by_documents_heuristics(top2vec_model: Top2Vec):
    # With and without returning the original documents
    num_to_examine = 5
    for return_docs in [True, False]:
        num_docs = 50
        get_docs = return_docs and top2vec_model.documents is not None
        # This should be the same as searching and not returning what was provided
        # Because this performs an ignore operation it is different than just
        # searching by an averaged vector in the similarity module
        for doc_num in range(num_docs):
            # going to search for close documents, make sure that things are as expected, and
            # say that at least the closest 3 documents in embedding space should be there

            closest_docs = find_similar_in_embedding(
                top2vec_model.document_vectors,
                positive_indices=[doc_num],
                cutoff_args=top2vec_model.cutoff_args,
            )
            # Going to test regardless of if we are configured to use
            # heuristics
            # this can be int or string
            doc_id = top2vec_model._get_document_ids([doc_num])[0]
            res = top2vec_model.search_documents_by_documents(
                [doc_id],
                num_docs,
                return_documents=return_docs,
                use_index=False,
            )
            res_docs, res_scores, res_indices = document_return_helper(
                top2vec_model, res, get_docs
            )
            assert not np.any(res_indices == doc_id)
            assert not np.any(closest_docs.indices == doc_num)
            to_examine = min(len(closest_docs.indices), num_to_examine)
            for x in range(to_examine):
                close_index = closest_docs.indices[x]
                close_id = top2vec_model._get_document_ids([close_index])[0]
                close_score = closest_docs.scores[x]
                assert np.any(res_indices == close_id)
                assert (
                    abs(
                        round(res_scores[x], ndigits=N_DECIMALS)
                        - round(close_score, ndigits=N_DECIMALS)
                    )
                    < 0.001
                )


@pytest.mark.parametrize("top2vec_model", models)
def test_similar_words(top2vec_model: Top2Vec):
    num_words = min(len(top2vec_model.vocab) - 1, 50)
    # This should return similar items that do NOT include the provided keywords
    for word_index in range(num_words):
        word = top2vec_model.vocab[word_index]
        word_vector = top2vec_model.word_vectors[word_index]
        words, scores = top2vec_model.similar_words(
            keywords=[word],
            keywords_neg=None,
            num_words=num_words,
            use_index=False,
        )
        assert len(words) <= num_words
        assert len(words) == len(scores)
        assert word not in set(words)

        # NOTE the cutoff heuristics will actually ignore
        # the provided indices when computing an elbow,
        if top2vec_model.use_cutoff_heuristics:
            expected_words, expected_scores = describe_closest_items(
                vectors=word_vector,
                embedding=top2vec_model.word_vectors,
                embedding_vocabulary=top2vec_model.vocab,
                topn=num_words,
                cutoff_args=top2vec_model.cutoff_args,
                ignore_indices=[word_index],
            )[0]
            # Make sure neither has the word in it
            assert word not in set(expected_words)
            assert set(words) == set(expected_words)
            assert compare_numpy_arrays(
                scores, expected_scores, round=True, print_verbose=False
            )

        # Because this is negative we shouldn't need to worry about
        # the initial item being returned.
        # Again, we have problems if our model says something which
        # is a complete opposite to our query is within the acceptable cutoff
        words, scores = top2vec_model.similar_words(
            keywords=[],
            keywords_neg=[word],
            num_words=num_words,
            use_index=False,
        )
        if words is not None and len(words) != 0:
            assert len(words) == len(scores)
            assert not np.any(words == word)
        if top2vec_model.use_cutoff_heuristics:
            expected_words, expected_scores = describe_closest_items(
                vectors=-word_vector,
                embedding=top2vec_model.word_vectors,
                embedding_vocabulary=top2vec_model.vocab,
                topn=num_words,
                cutoff_args=top2vec_model.cutoff_args,
                ignore_indices=[word_index],
            )[0]
            assert len(words) == len(expected_words)
            assert len(scores) == len(expected_scores)
            # order can be slightly different with these
            # if we have identical values
            assert set(words) == set(expected_words)
            assert compare_numpy_arrays(scores, expected_scores, round=True)


def test_cutoff_args_passthrough():
    # Take a single model and make sure that we get a value error when
    # given a bogus heuristic
    top2vec_model = top2vec_use_cutoff
    old_cutoff_args = top2vec_model.cutoff_args
    old_use_cutoff_heuristics = top2vec_model.use_cutoff_heuristics
    top2vec_model.cutoff_args = {"cutoff_heuristic": "A fake heuristic name"}
    with pytest.raises(ValueError):
        top2vec_model.similar_words(
            top2vec_model.vocab[:2], num_words=5, use_index=False
        )
    with pytest.raises(ValueError):
        top2vec_model._search_vectors_by_vector_heuristic(
            top2vec_model.word_vectors[[0, 1]],
            top2vec_model.word_vectors[0],
            topn=5,
            cutoff_args=top2vec_model.cutoff_args,
        )
    with pytest.raises(ValueError):
        top2vec_model.search_documents_by_vector(
            top2vec_model.word_vectors[0], num_docs=5, return_documents=True
        )
    with pytest.raises(ValueError):
        top2vec_model.search_documents_by_topic(0, num_docs=5, return_documents=True)
    with pytest.raises(ValueError):
        top2vec_model.search_documents_by_keywords(
            top2vec_model.vocab[:2], num_docs=5, return_documents=True
        )
    with pytest.raises(ValueError):
        top2vec_model.search_documents_by_documents(
            [0], num_docs=5, return_documents=True
        )
    with pytest.raises(ValueError):
        top2vec_model.search_words_by_vector(
            top2vec_model.word_vectors[0], num_words=5, use_index=False
        )
    with pytest.raises(ValueError):
        top2vec_model.search_words_by_vector(
            top2vec_model.word_vectors[0], num_words=5, use_index=False
        )
    with pytest.raises(ValueError):
        top2vec_model.search_topics_by_vector(
            top2vec_model.word_vectors[0], num_topics=5
        )
    with pytest.raises(ValueError):
        top2vec_model.search_topics(top2vec_model.vocab[:2], num_topics=1)

    top2vec_model.cutoff_args = old_cutoff_args
    top2vec_model.use_cutoff_heuristics = old_use_cutoff_heuristics


@pytest.mark.parametrize("top2vec_model", models)
@pytest.mark.parametrize("reduced", [True, False])
@pytest.mark.parametrize("index", [True, False])
def test_describe_topic(top2vec_model: Top2Vec, reduced: bool, index: bool):
    base_len = top2vec_model.max_topic_terms
    new_len = base_len * 2
    # Make sure we actually have a hierarchical reduction
    if reduced and top2vec_model.hierarchy is None:
        return
    if index and top2vec_model.word_index is None:
        return

    # Ensure above and below raises errors
    if reduced:
        num_topics = len(top2vec_model.topic_vectors_reduced)
    else:
        num_topics = len(top2vec_model.topic_vectors)

    with pytest.raises(ValueError):
        top2vec_model.describe_topic(-1, new_len, reduced=reduced, index=index)

    with pytest.raises(ValueError):
        top2vec_model.describe_topic(num_topics, new_len, reduced=reduced, index=index)

    for topic_num in range(num_topics):
        similar_words = top2vec_model.describe_topic(
            topic_num, num_words=new_len, reduced=reduced, index=index
        )
        if top2vec_model.use_cutoff_heuristics:
            assert len(similar_words.items) <= new_len
        else:
            assert len(similar_words.items) == new_len

        similar_words = top2vec_model.describe_topic(
            topic_num, num_words=None, reduced=reduced, index=index
        )
        if top2vec_model.use_cutoff_heuristics:
            assert len(similar_words.items) <= base_len
        else:
            assert len(similar_words.items) == base_len
