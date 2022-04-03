import pytest
from top2vec.Top2Vec import Top2Vec
from sklearn.datasets import fetch_20newsgroups
import numpy as np
import tempfile
import tensorflow_hub as hub

# get 20 newsgroups data
newsgroups_train = fetch_20newsgroups(subset='all', remove=('headers', 'footers', 'quotes'))
newsgroups_documents = newsgroups_train.data[0:2000]

# train top2vec model without doc_ids provided
top2vec = Top2Vec(documents=newsgroups_documents, speed="fast-learn", workers=8)

# train top2vec model with doc_ids provided
doc_ids = [str(num) for num in range(0, len(newsgroups_documents))]
top2vec_docids = Top2Vec(documents=newsgroups_documents, document_ids=doc_ids, speed="fast-learn", workers=8)

# train top2vec model without saving documents
top2vec_no_docs = Top2Vec(documents=newsgroups_documents, keep_documents=False, speed="fast-learn", workers=8)

# train top2vec model with corpus_file
top2vec_corpus_file = Top2Vec(documents=newsgroups_documents, use_corpus_file=True, speed="fast-learn", workers=8)

# test USE
top2vec_use = Top2Vec(documents=newsgroups_documents, embedding_model='universal-sentence-encoder')

# test USE with model embedding
top2vec_use_model_embedding = Top2Vec(documents=newsgroups_documents,
                                      embedding_model='universal-sentence-encoder',
                                      use_embedding_model_tokenizer=True)

# test USE-multilang
top2vec_use_multilang = Top2Vec(documents=newsgroups_documents,
                                embedding_model='universal-sentence-encoder-multilingual')

# test Sentence Transformer-multilang
top2vec_transformer_multilang = Top2Vec(documents=newsgroups_documents,
                                        embedding_model='distiluse-base-multilingual-cased')

# test Sentence Transformer with model embedding
top2vec_transformer_model_embedding = Top2Vec(documents=newsgroups_documents,
                                              embedding_model='distiluse-base-multilingual-cased',
                                              use_embedding_model_tokenizer=True)

top2vec_transformer_use_large = Top2Vec(documents=newsgroups_documents,
                                        embedding_model='universal-sentence-encoder-large',
                                        use_embedding_model_tokenizer=True,
                                        split_documents=True)

top2vec_transformer_use_multilang_large = Top2Vec(documents=newsgroups_documents,
                                                  embedding_model='universal-sentence-encoder-multilingual-large',
                                                  use_embedding_model_tokenizer=True,
                                                  split_documents=True,
                                                  document_chunker='random')

top2vec_transformer_sbert_l6 = Top2Vec(documents=newsgroups_documents,
                                       embedding_model='all-MiniLM-L6-v2',
                                       use_embedding_model_tokenizer=True,
                                       split_documents=True,
                                       document_chunker='sequential')

top2vec_transformer_sbert_l12 = Top2Vec(documents=newsgroups_documents,
                                        embedding_model='paraphrase-multilingual-MiniLM-L12-v2',
                                        use_embedding_model_tokenizer=True,
                                        split_documents=True,
                                        document_chunker='random'
                                        )

model = hub.load('https://tfhub.dev/google/universal-sentence-encoder/4')
top2vec_model_callable = Top2Vec(documents=newsgroups_documents,
                                 embedding_model=model,
                                 use_embedding_model_tokenizer=True,
                                 split_documents=True,
                                 document_chunker='random'
                                 )

top2vec_ngrams = Top2Vec(documents=newsgroups_documents,
                         speed="fast-learn",
                         ngram_vocab=True,
                         workers=8)

top2vec_use_ngrams = Top2Vec(documents=newsgroups_documents,
                             embedding_model='universal-sentence-encoder',
                             ngram_vocab=True)

models = [top2vec, top2vec_docids, top2vec_no_docs, top2vec_corpus_file,
          top2vec_use, top2vec_use_multilang, top2vec_transformer_multilang,
          top2vec_use_model_embedding, top2vec_transformer_model_embedding,
          top2vec_transformer_use_large,
          top2vec_transformer_use_multilang_large,
          top2vec_transformer_sbert_l6,
          top2vec_transformer_sbert_l12,
          top2vec_model_callable,
          top2vec_ngrams,
          top2vec_use_ngrams]


@pytest.mark.parametrize('top2vec_model', models)
def test_add_documents_original(top2vec_model):
    num_docs = top2vec_model.document_vectors.shape[0]

    docs_to_add = newsgroups_train.data[0:100]

    topic_count_sum = sum(top2vec_model.get_topic_sizes()[0])

    if top2vec_model.document_ids_provided is False:
        top2vec_model.add_documents(docs_to_add)
    else:
        doc_ids_new = [str(num) for num in range(2000, 2000 + len(docs_to_add))]
        top2vec_model.add_documents(docs_to_add, doc_ids_new)

    topic_count_sum_new = sum(top2vec_model.get_topic_sizes()[0])
    num_docs_new = top2vec_model.document_vectors.shape[0]

    assert topic_count_sum + len(docs_to_add) == topic_count_sum_new == num_docs + len(docs_to_add) \
           == num_docs_new == len(top2vec_model.doc_top)

    if top2vec_model.documents is not None:
        assert num_docs_new == len(top2vec_model.documents)


@pytest.mark.parametrize('top2vec_model', models)
def test_hierarchical_topic_reduction(top2vec_model):
    num_topics = top2vec_model.get_num_topics()

    if num_topics > 10:
        reduced_num = 10
    elif num_topics - 1 > 0:
        reduced_num = num_topics - 1

    hierarchy = top2vec_model.hierarchical_topic_reduction(reduced_num)

    assert len(hierarchy) == reduced_num == len(top2vec_model.topic_vectors_reduced)


@pytest.mark.parametrize('top2vec_model', models)
def test_add_documents_post_reduce(top2vec_model):
    docs_to_add = newsgroups_train.data[500:600]

    num_docs = top2vec_model.document_vectors.shape[0]
    topic_count_sum = sum(top2vec_model.get_topic_sizes()[0])
    topic_count_reduced_sum = sum(top2vec_model.get_topic_sizes(reduced=True)[0])

    if top2vec_model.document_ids_provided is False:
        top2vec_model.add_documents(docs_to_add)
    else:
        doc_ids_new = [str(num) for num in range(2100, 2100 + len(docs_to_add))]
        top2vec_model.add_documents(docs_to_add, doc_ids_new)

    topic_count_sum_new = sum(top2vec_model.get_topic_sizes()[0])
    topic_count_reduced_sum_new = sum(top2vec_model.get_topic_sizes(reduced=True)[0])

    num_docs_new = top2vec_model.document_vectors.shape[0]

    assert topic_count_sum + len(docs_to_add) == topic_count_sum_new == topic_count_reduced_sum + len(docs_to_add) \
           == topic_count_reduced_sum_new == num_docs + len(docs_to_add) == num_docs_new == len(top2vec_model.doc_top) \
           == len(top2vec_model.doc_top_reduced)

    if top2vec_model.documents is not None:
        assert num_docs_new == len(top2vec_model.documents)


@pytest.mark.parametrize('top2vec_model', models)
def test_delete_documents(top2vec_model):
    doc_ids_to_delete = list(range(500, 550))

    num_docs = top2vec_model.document_vectors.shape[0]
    topic_count_sum = sum(top2vec_model.get_topic_sizes()[0])
    topic_count_reduced_sum = sum(top2vec_model.get_topic_sizes(reduced=True)[0])

    if top2vec_model.document_ids_provided is False:
        top2vec_model.delete_documents(doc_ids=doc_ids_to_delete)
    else:
        doc_ids_to_delete = [str(doc_id) for doc_id in doc_ids_to_delete]
        top2vec_model.delete_documents(doc_ids=doc_ids_to_delete)

    topic_count_sum_new = sum(top2vec_model.get_topic_sizes()[0])
    topic_count_reduced_sum_new = sum(top2vec_model.get_topic_sizes(reduced=True)[0])
    num_docs_new = top2vec_model.document_vectors.shape[0]

    assert topic_count_sum - len(doc_ids_to_delete) == topic_count_sum_new == topic_count_reduced_sum - \
           len(doc_ids_to_delete) == topic_count_reduced_sum_new == num_docs - len(doc_ids_to_delete) \
           == num_docs_new == len(top2vec_model.doc_top) == len(top2vec_model.doc_top_reduced)

    if top2vec_model.documents is not None:
        assert num_docs_new == len(top2vec_model.documents)


@pytest.mark.parametrize('top2vec_model', models)
def test_get_topic_hierarchy(top2vec_model):
    hierarchy = top2vec_model.get_topic_hierarchy()

    assert len(hierarchy) == len(top2vec_model.topic_vectors_reduced)


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_get_num_topics(top2vec_model, reduced):
    # check that there are more than 0 topics
    assert top2vec_model.get_num_topics(reduced=reduced) > 0


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_get_topics(top2vec_model, reduced):
    num_topics = top2vec_model.get_num_topics(reduced=reduced)
    words, word_scores, topic_nums = top2vec_model.get_topics(reduced=reduced)

    # check that for each topic there are words, word_scores and topic_nums
    assert len(words) == len(word_scores) == len(topic_nums) == num_topics

    # check that for each word there is a score
    assert len(words[0]) == len(word_scores[0])

    # check that topics words are returned in decreasing order
    topic_words_scores = word_scores[0]
    assert all(topic_words_scores[i] >= topic_words_scores[i + 1] for i in range(len(topic_words_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_get_topic_size(top2vec_model, reduced):
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes(reduced=reduced)

    # check that topic sizes add up to number of documents
    assert sum(topic_sizes) == top2vec_model.document_vectors.shape[0]

    # check that topics are ordered decreasingly
    assert all(topic_sizes[i] >= topic_sizes[i + 1] for i in range(len(topic_sizes) - 1))


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_generate_topic_wordcloud(top2vec_model, reduced):
    # generate word cloud
    num_topics = top2vec_model.get_num_topics(reduced=reduced)
    top2vec_model.generate_topic_wordcloud(num_topics - 1, reduced=reduced)


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_search_documents_by_topic(top2vec_model, reduced):
    # get topic sizes
    topic_sizes, topic_nums = top2vec_model.get_topic_sizes(reduced=reduced)
    topic = topic_nums[0]
    num_docs = topic_sizes[0]

    # search documents by topic
    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_topic(topic, num_docs,
                                                                                           reduced=reduced)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_topic(topic, num_docs, reduced=reduced)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))

    # check that all documents returned are most similar to topic being searched
    document_indexes = [top2vec_model.doc_id2index[doc_id] for doc_id in document_ids]

    if reduced:
        doc_topics = set(np.argmax(
            np.inner(top2vec_model.document_vectors[document_indexes],
                     top2vec_model.topic_vectors_reduced), axis=1))
    else:
        doc_topics = set(np.argmax(
            np.inner(top2vec_model.document_vectors[document_indexes],
                     top2vec_model.topic_vectors), axis=1))
    assert len(doc_topics) == 1 and topic in doc_topics


@pytest.mark.parametrize('top2vec_model', models)
def test_search_documents_by_keywords(top2vec_model):
    keywords = top2vec_model.vocab
    keyword = keywords[-1]
    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_keywords(keywords=[keyword],
                                                                                              num_docs=num_docs)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_keywords(keywords=[keyword],
                                                                                   num_docs=num_docs)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_similar_words(top2vec_model):
    keywords = top2vec_model.vocab
    keyword = keywords[-1]
    num_words = 20

    words, word_scores = top2vec_model.similar_words(keywords=[keyword], num_words=num_words)

    # check that there is a score for each word
    assert len(words) == len(word_scores) == num_words

    # check that words are returned in decreasing order
    assert all(word_scores[i] >= word_scores[i + 1] for i in range(len(word_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
@pytest.mark.parametrize('reduced', [False, True])
def test_search_topics(top2vec_model, reduced):
    num_topics = top2vec_model.get_num_topics(reduced=reduced)
    keywords = top2vec_model.vocab
    keyword = keywords[-1]
    topic_words, word_scores, topic_scores, topic_nums = top2vec_model.search_topics(keywords=[keyword],
                                                                                     num_topics=num_topics,
                                                                                     reduced=reduced)
    # check that for each topic there are topic words, word scores, topic scores and score of topic
    assert len(topic_words) == len(word_scores) == len(topic_scores) == len(topic_nums) == num_topics

    # check that for each topic words have scores
    assert len(topic_words[0]) == len(word_scores[0])

    # check that topics are returned in decreasing order
    assert all(topic_scores[i] >= topic_scores[i + 1] for i in range(len(topic_scores) - 1))

    # check that topics words are returned in decreasing order
    topic_words_scores = word_scores[0]
    assert all(topic_words_scores[i] >= topic_words_scores[i + 1] for i in range(len(topic_words_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_search_document_by_documents(top2vec_model):
    doc_id = top2vec_model.document_ids[0]

    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_documents(doc_ids=[doc_id],
                                                                                               num_docs=num_docs)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_documents(doc_ids=[doc_id],
                                                                                    num_docs=num_docs)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_get_documents_topics(top2vec_model):
    doc_ids_get = top2vec_model.document_ids[[0, 5]]

    if top2vec_model.hierarchy is not None:
        doc_topics, doc_dist, topic_words, topic_word_scores = top2vec_model.get_documents_topics(doc_ids=doc_ids_get,
                                                                                                  reduced=True)
    else:
        doc_topics, doc_dist, topic_words, topic_word_scores = top2vec_model.get_documents_topics(doc_ids=doc_ids_get)

    assert len(doc_topics) == len(doc_dist) == len(topic_words) == len(topic_word_scores) == len(doc_ids_get)


@pytest.mark.parametrize('top2vec_model', models)
def test_get_documents_topics_multiple(top2vec_model):
    doc_ids_get = top2vec_model.document_ids[[0, 1, 5]]
    num_topics = 2

    if top2vec_model.hierarchy is not None:
        doc_topics, doc_dist, topic_words, topic_word_scores = top2vec_model.get_documents_topics(doc_ids=doc_ids_get,
                                                                                                  reduced=True,
                                                                                                  num_topics=num_topics)

        actual_number_topics = top2vec_model.get_num_topics(reduced=True)

    else:
        doc_topics, doc_dist, topic_words, topic_word_scores = top2vec_model.get_documents_topics(doc_ids=doc_ids_get,
                                                                                                  num_topics=num_topics)

        actual_number_topics = top2vec_model.get_num_topics(reduced=False)

    assert len(doc_topics) == len(doc_dist) == len(topic_words) == len(topic_word_scores) == len(doc_ids_get)

    if num_topics <= actual_number_topics:
        assert doc_topics.shape[1] == num_topics
        assert doc_dist.shape[1] == num_topics
        assert topic_words.shape[1] == num_topics
        assert topic_word_scores.shape[1] == num_topics


@pytest.mark.parametrize('top2vec_model', models)
def test_search_documents_by_vector(top2vec_model):
    document_vectors = top2vec_model.document_vectors
    top2vec_model.search_documents_by_vector(vector=document_vectors[0], num_docs=10)

    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_vector(vector=document_vectors[0],
                                                                                            num_docs=num_docs)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_vector(vector=document_vectors[0],
                                                                                 num_docs=num_docs)
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_index_documents(top2vec_model):
    top2vec_model.index_document_vectors()
    assert top2vec_model.document_vectors.shape[1] <= top2vec_model.document_index.get_max_elements()


@pytest.mark.parametrize('top2vec_model', models)
def test_search_documents_by_vector_index(top2vec_model):
    document_vectors = top2vec_model.document_vectors
    top2vec_model.search_documents_by_vector(vector=document_vectors[0], num_docs=10)

    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_vector(vector=document_vectors[0],
                                                                                            num_docs=num_docs,
                                                                                            use_index=True)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_vector(vector=document_vectors[0],
                                                                                 num_docs=num_docs,
                                                                                 use_index=True)
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_search_documents_by_keywords_index(top2vec_model):
    keywords = top2vec_model.vocab
    keyword = keywords[-1]
    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_keywords(keywords=[keyword],
                                                                                              num_docs=num_docs,
                                                                                              use_index=True)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_keywords(keywords=[keyword],
                                                                                   num_docs=num_docs,
                                                                                   use_index=True)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_search_document_by_documents_index(top2vec_model):
    doc_id = top2vec_model.document_ids[0]

    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.search_documents_by_documents(doc_ids=[doc_id],
                                                                                               num_docs=num_docs,
                                                                                               use_index=True)
    else:
        document_scores, document_ids = top2vec_model.search_documents_by_documents(doc_ids=[doc_id],
                                                                                    num_docs=num_docs,
                                                                                    use_index=True)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_search_words_by_vector(top2vec_model):
    word_vectors = top2vec_model.word_vectors
    top2vec_model.search_words_by_vector(vector=word_vectors[0], num_words=10)

    num_words = 10

    words, word_scores = top2vec_model.search_words_by_vector(vector=word_vectors[0],
                                                              num_words=num_words)

    # check that there is a score for each word
    assert len(words) == len(word_scores) == num_words

    # check that words are returned in decreasing order
    assert all(word_scores[i] >= word_scores[i + 1] for i in range(len(word_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_index_words(top2vec_model):
    top2vec_model.index_word_vectors()
    assert top2vec_model.word_vectors.shape[1] <= top2vec_model.word_index.get_max_elements()


@pytest.mark.parametrize('top2vec_model', models)
def test_similar_words_index(top2vec_model):
    keywords = top2vec_model.vocab
    keyword = keywords[-1]
    num_words = 20

    words, word_scores = top2vec_model.similar_words(keywords=[keyword], num_words=num_words, use_index=True)

    # check that there is a score for each word
    assert len(words) == len(word_scores) == num_words

    # check that words are returned in decreasing order
    assert all(word_scores[i] >= word_scores[i + 1] for i in range(len(word_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_save_load(top2vec_model):
    if top2vec_model.embedding_model == "custom":
        model = top2vec_model.embed
        temp = tempfile.NamedTemporaryFile(mode='w+b')
        top2vec_model.save(temp.name)
        Top2Vec.load(temp.name)
        temp.close()
        top2vec_model.set_embedding_model(model)

    else:
        temp = tempfile.NamedTemporaryFile(mode='w+b')
        top2vec_model.save(temp.name)
        Top2Vec.load(temp.name)
        temp.close()


@pytest.mark.parametrize('top2vec_model', models)
def test_query_documents(top2vec_model):
    num_docs = 10

    if top2vec_model.documents is not None:
        documents, document_scores, document_ids = top2vec_model.query_documents(query="what is the meaning of life?",
                                                                                 num_docs=num_docs)
    else:
        document_scores, document_ids = top2vec_model.query_documents(query="what is the meaning of life?",
                                                                      num_docs=num_docs)

    # check that for each document there is a score and number
    if top2vec_model.documents is not None:
        assert len(documents) == len(document_scores) == len(document_ids) == num_docs
    else:
        assert len(document_scores) == len(document_ids) == num_docs

    # check that documents are returned in decreasing order
    assert all(document_scores[i] >= document_scores[i + 1] for i in range(len(document_scores) - 1))


@pytest.mark.parametrize('top2vec_model', models)
def test_query_topics(top2vec_model):
    num_topics = top2vec_model.get_num_topics()
    topic_words, word_scores, topic_scores, topic_nums = top2vec_model.query_topics(query="what is the "
                                                                                          "meaning of life?",
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
