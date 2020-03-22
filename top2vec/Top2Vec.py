# Author: Dimo Angelov
#
# License: BSD 3 clause
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
import umap
import hdbscan
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity


class Top2Vec:
    """
    Topic2Vector

    Creates jointly embedded topic, document and word vectors.


    Parameters
    ----------
    documents: list of str
        Input corpus, should be a list of strings.

    speed: string (optional, default 'fast-learn')
        This parameter will determine how fast the model takes to train. The
        fast-learn option is the fastest and will generate the lowest quality
        vectors. The learn option will learn better quality vectors but take
        a longer time to train. The deep-learn option will learn the best quality
        vectors but will take significant time to train. The valid string speed
        options are:
            * fast-learn
            * learn
            * deep-learn

    workers: int (optional)
        The amount of worker threads to be used in training the model. Larger
        amount will lead to faster training.

    Methods
    -------
    """

    def __init__(self, documents, speed="fast-learn", workers=None):
        """
        Parameters
        ----------
        documents: list of str
            Input corpus, should be a list of strings.

        speed: string (optional, default 'fast-learn')
            This parameter will determine how fast the model takes to train. The
            fast-learn option is the fastest and will generate the lowest quality
            vectors. The learn option will learn better quality vectors but take
            a longer time to train. The deep-learn option will learn the best quality
            vectors but will take significant time to train. The valid string speed
            options are:
                * fast-learn
                * learn
                * deep-learn

        workers: int (optional)
            The amount of worker threads to be used in training the model. Larger
            amount will lead to faster training.
        """
        # validate inputs
        if speed == "fast-learn":
            hs = 0
            negative = 5
            epochs = 40
        elif speed == "learn":
            hs = 1
            negative = 0
            epochs = 40
        elif speed == "deep-learn":
            hs = 1
            negative = 0
            epochs = 400
        else:
            raise ValueError("speed parameter needs to be one of: fast-learn, learn or deep-learn")

        if workers is None:
            pass
        elif isinstance(workers, int):
            pass
        else:
            raise ValueError("workers needs to be an int")

        self.documents = documents

        # preprocess documents for training - tokenize and remove too long/short words
        train_corpus = [TaggedDocument(simple_preprocess(doc), [i]) for i, doc in enumerate(documents)]

        # create documents and word embeddings with doc2vec
        if workers is None:
            self.model = Doc2Vec(documents=train_corpus, vector_size=300, min_count=50, window=15,
                                 sample=10e-5, negative=negative, hs=hs, epochs=epochs, dm=0,
                                 dbow_words=1)
        else:
            self.model = Doc2Vec(documents=train_corpus, vector_size=300, min_count=50, window=15,
                                 sample=10e-5, negative=negative, hs=hs, workers=workers, epochs=epochs, dm=0,
                                 dbow_words=1)

        # create 5D embeddings of documents
        docvecs = np.vstack([self.model.docvecs[i] for i in range(self.model.docvecs.count)])
        umap_model = umap.UMAP(n_neighbors=15, n_components=5, metric='cosine').fit(docvecs)

        # find dense areas of document vectors
        cluster = hdbscan.HDBSCAN(min_cluster_size=15, metric='euclidean', cluster_selection_method='eom').fit(
            umap_model.embedding_)

        cluster_labels = pd.Series(cluster.labels_)

        # generate topic vectors from dense areas of documents
        self.topic_vectors = []
        self.topic_words = []
        self.topic_word_scores = []

        unique_labels = list(set(cluster_labels))
        unique_labels.remove(-1)

        for label in unique_labels:

            # find centroid of dense document cluster
            topic_vector = [0] * 300
            cluster_vec_indices = cluster_labels[cluster_labels == label].index.tolist()
            for vec_index in cluster_vec_indices:
                topic_vector = topic_vector + self.model.docvecs[vec_index]
            topic_vector = topic_vector / len(cluster_vec_indices)
            self.topic_vectors.append(topic_vector)

            # find closest word vectors to topic vector
            sim_words = self.model.most_similar(positive=[topic_vector], topn=50)
            self.topic_words.append([word[0] for word in sim_words])
            self.topic_word_scores.append([round(word[1], 4) for word in sim_words])

    def _validate_num_docs(self, num_docs):
        document_count = len(self.documents)
        if num_docs > document_count:
            raise ValueError(f"num_docs cannot exceed the number of topics: {document_count}")

    def _validate_num_topics(self, num_topics):
        topic_count = len(self.topic_vectors)
        if num_topics > topic_count:
            raise ValueError(f"num_topics cannot exceed the number of topics: {topic_count}")

    def _validate_topic_num(self, topic_num):
        topic_count = len(self.topic_vectors)-1
        if topic_num > topic_count:
            raise ValueError(f"Invalid topic number: valid topics numbers are 0 to {topic_count}")

    def _validate_doc_num(self, doc_num):
        document_count = len(self.documents)-1
        if doc_num > document_count:
            raise ValueError(f"Invalid topic number: valid topic numbers are 0 to {document_count}")

    def _validate_keywords(self, keywords, keywords_neg):
        for word in keywords+keywords_neg:
            if word not in self.model.wv.vocab:
                raise ValueError(f"'{word}' has not been learned by the model so it cannot be searched")

    def get_num_topics(self):
        """
        Get number of topics.

        This is the number of topics Top2Vec has found in the data.

        Returns
        -------
        num_topics: int
        """
        return len(self.topic_vectors)

    def get_topics(self, num_topics):
        """
        Get number of specified topics.

        Each topic will consist of the top 50 semantically similar words
        to the topic. These are the 50 words closest to topic vector
        along with cosine similarity of each word from vector. The
        higher the score the more relevant the word is to the topic.

        Parameters
        ----------
        num_topics: int
            Number of topics to return.

        Returns
        -------
        topics_words: list, shape (n_topics, 50)
            For each topic the top 50 words are returned, in order
            of semantic similarity to topic.
            Example:
                [['data', 'deep', 'learning' ... 'artificial'],          <Topic 0>
                 ['environment', 'warming', 'climate ... 'temperature']  <Topic 1>
                 ...]

        word_scores: list, shape (n_topics, 50)
            For each topic the cosine similarity scores of the
            top 50 words to the topic are returned.
            Example:
                [[0.7132, 0.6473, 0.5700 ... 0.3455],  <Topic 0>
                 [0.7818', 0.7671, 0.7603 ... 0.6769]  <Topic 1>
                 ...]

        topic_nums: list of int
            The unique index of every topic will be returned.
        """

        self._validate_num_topics(num_topics)

        return self.topic_words[0:num_topics], self.topic_word_scores[0:num_topics], list(range(0, num_topics))

    def search_documents_by_topic(self, topic_num, num_docs):
        """
        Get the most semantically similar documents to the topic.

        These are the documents closest to the topic vector. Documents are ordered
        by proximity to the topic vector. Successive documents in the list are
        less semantically similar to the topic.

        Parameters
        ----------
        topic_num: int
            The topic number to search.

        num_docs: int
            Number of documents to return.

        Returns
        -------
        documents: list of str
            The documents in a list, the most similar are first.

        doc_scores: float
            Semantic similarity of document to topic. The cosine similarity of the
            document and topic vector.

        doc_nums: list of int
            Indexes of documents in the input corpus of documents.
        """

        self._validate_num_docs(num_docs)
        self._validate_topic_num(topic_num)

        sim_docs = self.model.docvecs.most_similar(positive=[self.topic_vectors[topic_num]], topn=num_docs)
        doc_nums = [doc[0] for doc in sim_docs]
        doc_scores = [round(doc[1], 4) for doc in sim_docs]
        documents = list(itemgetter(*doc_nums)(self.documents))

        return documents, doc_scores, doc_nums

    def search_documents_by_keyword(self, keywords, num_docs, keywords_neg=[]):
        """
        Semantic search of documents using keywords.

        The most semantically similar documents to the combination of the keywords
        will be returned. If negative keywords are provided, the documents will be
        semantically dissimilar to those words. Too many keywords or certain
        combinations of words may give strange results. This method finds an average
        vector(negative keywords are subtracted) of all the keyword vectors and
        returns the documents closest to the resulting vector.

        Parameters
        ----------
        keywords: list of str
            List of positive keywords being used for search of semantically similar
            documents.

        keywords_neg: list of str (Optional)
            List of negative keywords being used for search of semantically dissimilar
            documents.

        num_docs: int
            Number of documents to return.

        Returns
        -------
        documents: list of str
            The documents in a list, the most similar are first.

        doc_scores: list of float
            Semantic similarity of document to keywords. The cosine similarity of the
            document and average of keyword vectors.

        doc_nums: list of int
            Indexes of documents in the input corpus of documents.
        """
        self._validate_num_docs(num_docs)
        self._validate_keywords(keywords, keywords_neg)

        word_vecs = [self.model[word] for word in keywords]
        neg_word_vecs = [self.model[word] for word in keywords_neg]
        sim_docs = self.model.docvecs.most_similar(positive=word_vecs, negative=neg_word_vecs, topn=num_docs)
        doc_nums = [doc[0] for doc in sim_docs]
        doc_scores = [round(doc[1], 4) for doc in sim_docs]
        documents = list(itemgetter(*doc_nums)(self.documents))

        return documents, doc_scores, doc_nums

    def similar_words(self, keywords, num_words, keywords_neg=[]):
        """
        Semantic similarity search of words.

        The most semantically similar word to the combination of the keywords
        will be returned. If negative keywords are provided, the words will be
        semantically dissimilar to those words. Too many keywords or certain
        combinations of words may give strange results. This method finds an average
        vector(negative keywords are subtracted) of all the keyword vectors and
        returns the words closest to the resulting vector.

        Parameters
        ----------
        keywords: list of str
            List of positive keywords being used for search of semantically similar
            words.

        keywords_neg: list of str
            List of negative keywords being used for search of semantically dissimilar
            words.

        num_words: int
            Number of words to return.


        Returns
        -------
        words: list of str
            The words in a list, the most similar are first.

        word_scores: list of float
            Semantic similarity of word to keywords. The cosine similarity of the
            word and average of keyword vectors.
        """
        self._validate_keywords(keywords, keywords_neg)

        word_vecs = [self.model[word] for word in keywords]
        neg_word_vecs = [self.model[word] for word in keywords_neg]
        sim_words = self.model.most_similar(positive=word_vecs, negative=neg_word_vecs, topn=num_words)
        words = [word[0] for word in sim_words]
        word_scores = [round(word[1],4) for word in sim_words]

        return words, word_scores

    def search_topics(self, keywords, num_topics, keywords_neg=[]):
        """
        Semantic search of topics using keywords.

        The most semantically similar topics to the combination of the keywords
        will be returned. If negative keywords are provided, the topics will be
        semantically dissimilar to those words. Topics will be ordered by
        decreasing similarity to the keywords. Too many keywords or certain
        combinations of words may give strange results. This method finds an average
        vector(negative keywords are subtracted) of all the keyword vectors and
        returns the topics closest to the resulting vector.

        Parameters
        ----------
        keywords: list of str
            List of positive keywords being used for search of semantically similar
            documents.

        keywords_neg: (Optional) list of str
            List of negative keywords being used for search of semantically dissimilar
            documents.

        num_topics: int
            Number of documents to return.

        Returns
        -------
        topics_words: list, shape (n_topics, 50)
            For each topic the top 50 words are returned, in order of semantic similarity to topic.
            Example:
                [['data', 'deep', 'learning' ... 'artificial'],             <Topic 0>
                 ['environment', 'warming', 'climate ... 'temperature']     <Topic 1>
                 ...]

        word_scores: list, shape (n_topics, 50)
            For each topic the cosine similarity scores of the top 50 words to the topic are returned.
            Example:
                [[0.7132, 0.6473, 0.5700 ... 0.3455],     <Topic 0>
                 [0.7818', 0.7671, 0.7603 ... 0.6769]     <Topic 1>
                 ...]

        topic_scores: list of float
            For each topic the cosine similarity to the search keywords will be returned.

        topic_nums: list of int
            The unique index of every topic will be returned.
        """
        self._validate_num_topics(num_topics)
        self._validate_keywords(keywords, keywords_neg)

        word_vecs = [self.model[word] for word in keywords]
        neg_word_vecs = [self.model[word] for word in keywords_neg]

        combined_vector = [0] * 300

        for word_vec in word_vecs:
            combined_vector += word_vec

        for word_vec in neg_word_vecs:
            combined_vector -= word_vec

        combined_vector /= (len(word_vecs) + len(neg_word_vecs))

        topic_ranks = [topic[0] for topic in cosine_similarity(self.topic_vectors, combined_vector.reshape(1, -1))]
        topic_nums = list(np.argsort(topic_ranks)[-num_topics:])
        topic_nums.reverse()

        topic_words = [self.topic_words[topic] for topic in topic_nums]
        word_scores = [self.topic_word_scores[topic] for topic in topic_nums]
        topic_scores = [round(topic_ranks[topic], 4) for topic in topic_nums]

        return topic_words, word_scores, topic_scores, topic_nums

    def search_documents_by_document(self, doc_num, num_docs):
        """
        Semantic similarity search of words.

        The most semantically similar documents to the document provided
        will be returned. This method finds the closest document vectors
        to the provided document vector.

        Parameters
        ----------
        doc_num: int
            Index of document in the input corpus of documents.

        num_docs: int
            Number of documents to return.

        Returns
        -------
        documents: list of str
            The documents in a list, the most similar are first.

        doc_scores: list of float
            Semantic similarity of document to keywords. The cosine similarity of the
            document and average of keyword vectors.

        doc_nums: list of int
            Indexes of documents in the input corpus of documents.
        """
        self._validate_num_docs(num_docs)
        self._validate_doc_num(doc_num)

        message_vec = self.model.docvecs[doc_num]
        sim_docs = self.model.docvecs.most_similar(positive=[message_vec], topn=num_docs)
        doc_nums = [doc[0] for doc in sim_docs]
        doc_scores = [round(doc[1], 4) for doc in sim_docs]
        documents = list(itemgetter(*doc_nums)(self.documents))

        return documents, doc_scores, doc_nums
