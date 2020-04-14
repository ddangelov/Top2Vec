import pytest
from top2vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# get 20 newsgroups data
newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
newsgroups_documents = newsgroups_train.data[0:1000]

# train top2vec model without doc_ids provided
top2vec = Top2Vec(documents=newsgroups_documents, speed="fast-learn", workers=8)

# train top2vec mode with doc_ids provided
doc_ids = [str(num) for num in range(0, len(newsgroups_documents))]
top2vec_docids = Top2Vec(documents=newsgroups_documents, document_ids=doc_ids, speed="fast-learn", workers=8)


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_get_num_topics(top2vec_model):
    # check that there are more than 0 topics
    assert top2vec_model.get_num_topics() > 0


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_get_topics(top2vec_model):
    num_topics = top2vec_model.get_num_topics()
    words, word_scores, topic_nums = top2vec_model.get_topics(num_topics)

    # check that for each topic there are words, word_scores and topic_nums
    assert len(words) == len(word_scores) == len(topic_nums) == num_topics

    # check that for each word there is a score
    assert len(words[0]) == len(word_scores[0])

    # check that topics words are returned in decreasing order
    topic_words_scores = word_scores[0]
    assert all(topic_words_scores[i] >= topic_words_scores[i + 1] for i in range(len(topic_words_scores) - 1))


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_get_topic_size(top2vec_model):
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()

    # check that topic sizes add up to number of documents
    assert sum(topic_sizes) == len(top2vec_model.documents)

    # check that topics are ordered decreasingly
    assert all(topic_sizes[i] >= topic_sizes[i + 1] for i in range(len(topic_sizes) - 1))


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_generate_topic_wordcloud(top2vec_model):
    # generate word cloud
    num_topics = top2vec_model.get_num_topics()
    top2vec_model.generate_topic_wordcloud(num_topics - 1)


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_search_documents_by_topic(top2vec_model):
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes()
    topic = topic_nums[0]
    num_docs = topic_sizes[0]
    documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic, num_docs)

    # check that for each document there is a score and number
    assert len(documents) == len(document_scores) == len(document_ids) == num_docs

    # check that number of documents returned for topic matches topic size
    assert len(documents) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))

    # check that all documents returned are most similar to topic being searched
    if top2vec_model.document_ids is not None:
        document_indexes = [top2vec_model.doc_id2index[doc_id] for doc_id in document_ids]
    else:
        document_indexes = document_ids

    doc_topics = set(np.argmax(
        cosine_similarity(top2vec_model.model.docvecs.vectors_docs[document_indexes],
                          top2vec_model.topic_vectors), axis=1))
    assert len(doc_topics) == 1 and topic in doc_topics


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_search_documents_by_keywords(top2vec_model):
    keywords = list(top2vec_model.model.wv.vocab.keys())
    keyword = keywords[-1]
    num_docs = 10

    documents, document_scores, document_ids = top2vec_model.search_documents_by_keywords(keywords=[keyword],
                                                                                          num_docs=num_docs)
    # check that for each document there is a score and number
    assert len(documents) == len(document_scores) == len(document_ids) == num_docs

    # check that correct number of documents is returned
    assert len(documents) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_similar_words(top2vec_model):
    keywords = list(top2vec_model.model.wv.vocab.keys())
    keyword = keywords[-1]
    num_words = 20

    words, word_scores = top2vec_model.similar_words(keywords=[keyword], num_words=num_words)

    # check that there is a score for each word
    assert len(words) == len(word_scores) == num_words

    # check that words are returned in decreasing order
    assert all(word_scores[i] >= word_scores[i + 1] for i in range(len(word_scores) - 1))


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_search_topics(top2vec_model):
    num_topics = top2vec_model.get_num_topics()
    keywords = list(top2vec_model.model.wv.vocab.keys())
    keyword = keywords[-1]
    topic_words, word_scores, topic_scores, topic_nums = top2vec_model.search_topics(keywords=[keyword],
                                                                                     num_topics=num_topics)

    # check that for each topic there are topic words, word scores, topic scores and score of topic
    assert len(topic_words) == len(word_scores) == len(topic_scores) == len(topic_nums) == num_topics

    # check that for each topic words have scores
    assert len(topic_words[0]) == len(word_scores[0])

    # check that topics are returned in decreasing order
    assert all(topic_scores[i] >= topic_scores[i + 1] for i in range(len(topic_scores) - 1))

    # check that topics words are returned in decreasing order
    topic_words_scores = word_scores[0]
    assert all(topic_words_scores[i] >= topic_words_scores[i + 1] for i in range(len(topic_words_scores) - 1))


@pytest.mark.parametrize('top2vec_model', [top2vec, top2vec_docids], ids=["top2vec_regular", "top2vec_doc_ids"])
def test_search_document_by_documents(top2vec_model):

    if top2vec_model.document_ids is not None:
        doc_id = top2vec_model.document_ids[0]
    else:
        doc_id = 0

    num_docs = 10

    documents, document_scores, document_ids = top2vec_model.search_documents_by_documents(doc_ids=[doc_id],
                                                                                           num_docs=num_docs)

    # check that for each document there is a score and number
    assert len(documents) == len(document_scores) == len(document_ids) == num_docs

    # check that correct number of documents is returned
    assert len(documents) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))
