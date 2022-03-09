from audioop import reverse
import pytest
from top2vec.Top2Vec import Top2Vec
from top2vec.similarity import (
    describe_closest_items,
    find_closest_items,
    generate_similarity_matrix,
    generate_csr_similarity_matrix,
    SimilarVectorIndices,
    SimilarItems,
)
import gensim
from sklearn.datasets import fetch_20newsgroups
import numpy as np

N_DECIMALS = 4


def compare_numpy_arrays(array_a, array_b, round=False):
    if array_a is None and array_b is None:
        return True
    elif array_a is None or array_b is None:
        print("One is None")
        return False
    # What about our size?
    if array_a.size == 0 and array_b.size == 0:
        return True
    elif array_a.size == 0 or array_b.size == 0:
        print("Unequal size")
        return False
    if array_a.shape != array_b.shape:
        print("Unequal shape")
        return False
    # Thanks a bunch, floating point numbers
    if round:
        # Have some items which are .0001 off
        diff = np.abs(
            array_a.round(decimals=N_DECIMALS) - array_b.round(decimals=N_DECIMALS)
        )
        if diff[diff > 0.001].any():
            print("Arrays more different than 0.001")
            print(diff)
            return False
        else:
            return True
    else:
        return (array_a == array_b).all()


# ensure consistent sorting
def sort_value_scores(iterable):
    by_values = sorted(iterable, key=lambda x: x[0], reverse=True)
    return sorted(by_values, key=lambda x: round(x[1], N_DECIMALS), reverse=True)


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

models = [
    top2vec_docids_cutoff,
    top2vec_use_model_embedding_cutoff,
    top2vec_use_multilang_cutoff,
    top2vec_use_cutoff,
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
            assert scores.size < topn
            assert set(expected_words) == set(words)
            assert compare_numpy_arrays(expected_scores, scores, round=True)
        else:
            assert scores.size == topn


@pytest.mark.parametrize("top2vec_model", models)
def test_document_descriptions(top2vec_model: Top2Vec):
    # Make sure we don't run out of memory
    maxDocs = 50
    topn = 1000
    document_descriptions = describe_closest_items(
        top2vec_model.document_vectors[:maxDocs],
        top2vec_model.word_vectors,
        top2vec_model.vocab,
        topn=topn,
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


@pytest.mark.parametrize("top2vec_model", models)
def test_document_topic_composition(top2vec_model: Top2Vec):
    # Theory is that MOST documents should be composed of a single topic
    # However in this case our USE has almost 100 topics despite there
    # being only 20 newsgroups. Therefore there is some duplication
    topn = 100
    document_topics = find_closest_items(
        top2vec_model.document_vectors, top2vec_model.topic_vectors, topn=topn
    )
    num_topics_per_doc = [len(scores) for indices, scores in document_topics]
    for num_topics in num_topics_per_doc:
        assert num_topics <= topn

    # Now let's see if the doc x topic matrix looks good
    docTopicMatrix = generate_similarity_matrix(
        top2vec_model.document_vectors, top2vec_model.topic_vectors, topn=topn
    )
    numZeroes = np.count_nonzero(docTopicMatrix == 0)
    sparsity = numZeroes / (docTopicMatrix.size)
    # Our sparsity should be high
    # It looks like we've got some weirdness where a few of the items
    # have a TON of topics
    assert sparsity > 0.8

    sparse_matrix = generate_csr_similarity_matrix(
        top2vec_model.document_vectors, top2vec_model.topic_vectors, topn=topn
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
    assert top2vec_use_model_embedding_cutoff.get_num_topics() == 80
    topn = 1000
    topic_descriptions = describe_closest_items(
        top2vec_use_model_embedding_cutoff.topic_vectors,
        top2vec_use_model_embedding_cutoff.word_vectors,
        top2vec_use_model_embedding_cutoff.vocab,
        topn=topn,
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
    assert topic_nums.size < 80

    (
        topics_words,
        word_scores,
        topic_scores,
        topic_nums,
    ) = top2vec_use_model_embedding_cutoff.query_topics("spacecraft jpl", num_topics=80)
    assert space_topic_num in topic_nums
    assert topic_nums.size < 80
