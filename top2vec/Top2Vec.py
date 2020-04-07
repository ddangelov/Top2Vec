# Author: Dimo Angelov
#
# License: BSD 3 clause
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import umap
import hdbscan
from operator import itemgetter
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.cluster import dbscan


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

        self.documents = list(documents)

        # preprocess documents for training - tokenize and remove too long/short words
        train_corpus = [TaggedDocument(simple_preprocess(strip_tags(doc), deacc=True), [i])
                        for i, doc in enumerate(documents)]

        # create documents and word embeddings with doc2vec
        if workers is None:
            self.model = Doc2Vec(documents=train_corpus,
                                 vector_size=300,
                                 min_count=50, window=15,
                                 sample=1e-5,
                                 negative=negative,
                                 hs=hs,
                                 epochs=epochs,
                                 dm=0,
                                 dbow_words=1)
        else:
            self.model = Doc2Vec(documents=train_corpus,
                                 vector_size=300,
                                 min_count=50,
                                 window=15,
                                 sample=1e-5,
                                 negative=negative,
                                 hs=hs,
                                 workers=workers,
                                 epochs=epochs,
                                 dm=0,
                                 dbow_words=1)

        # create 5D embeddings of documents
        umap_model = umap.UMAP(n_neighbors=15,
                               n_components=5,
                               metric='cosine').fit(self.model.docvecs.vectors_docs)

        # find dense areas of document vectors
        cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  metric='euclidean',
                                  cluster_selection_method='eom').fit(umap_model.embedding_)

        # calculate topic vectors from dense areas of documents
        self._create_topic_vectors(cluster.labels_)

        # deduplicate topics
        self._deduplicate_topics()

        # calculate topic sizes and index nearest topic for each document
        self._calculate_topic_sizes()

        # find topic words and scores
        self._find_topic_words_scores()

    def _create_topic_vectors(self, cluster_labels):

        unique_labels = set(cluster_labels)
        if -1 in unique_labels:
            unique_labels.remove(-1)
        self.topic_vectors = np.vstack([self.model.docvecs.vectors_docs[np.where(cluster_labels == label)]
                                       .mean(axis=0) for label in unique_labels])

    def _deduplicate_topics(self):
        core_samples, labels = dbscan(X=self.topic_vectors,
                                      eps=0.1,
                                      min_samples=2,
                                      metric="cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:
            duplicate_clusters.remove(-1)

            # unique topics
            unique_topics = self.topic_vectors[np.where(labels == -1)]

            # merge duplicate topics
            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self.topic_vectors[np.where(labels == unique_label)]
                        .mean(axis=0)])

            self.topic_vectors = unique_topics

    def _calculate_topic_sizes(self):
        # calculate topic size
        doc_top_sim = cosine_similarity(self.model.docvecs.vectors_docs, self.topic_vectors)
        self.topic_sizes = pd.Series(np.argmax(doc_top_sim, axis=1)).value_counts()

        # re-order topic vectors by size
        self.topic_vectors = self.topic_vectors[self.topic_sizes.index]
        doc_top_sim = doc_top_sim[:, self.topic_sizes.index]
        self.topic_sizes.reset_index(drop=True, inplace=True)

        # find nearest topic for each document and distance to topic
        self.doc_dist = np.max(doc_top_sim, axis=1)
        self.doc_top = np.argmax(doc_top_sim, axis=1)

    def _find_topic_words_scores(self):
        self.topic_words = []
        self.topic_word_scores = []
        np.apply_along_axis(self._generate_topic_words_scores, axis=1, arr=self.topic_vectors)

    def _generate_topic_words_scores(self, topic_vector):
        sim_words = self.model.wv.most_similar(positive=[topic_vector], topn=50)
        self.topic_words.append([word[0] for word in sim_words])
        self.topic_word_scores.append([round(word[1], 4) for word in sim_words])

    def save(self, file):
        """
        Saves the current model to the specified file.

        Parameters
        ----------
        file: str
            File where model will be saved.
        """
        dump(self, file)

    @classmethod
    def load(cls, file):
        """

        Load a pre-trained model from the specified file.

        Parameters
        ----------
        file: str
            File where model will be loaded from.
        """
        return load(file)

    @staticmethod
    def _less_than_zero(num, var_name):
        if num < 0:
            raise ValueError(f"{var_name} cannot be less than 0.")

    def _validate_num_docs(self, num_docs):
        self._less_than_zero(num_docs, "num_docs")
        document_count = len(self.documents)
        if num_docs > document_count:
            raise ValueError(f"num_docs cannot exceed the number of documents: {document_count}")

    def _validate_num_topics(self, num_topics):
        self._less_than_zero(num_topics, "num_topics")
        topic_count = len(self.topic_vectors)
        if num_topics > topic_count:
            raise ValueError(f"num_topics cannot exceed the number of topics: {topic_count}")

    def _validate_topic_num(self, topic_num):
        self._less_than_zero(topic_num, "topic_num")
        topic_count = len(self.topic_vectors) - 1
        if topic_num > topic_count:
            raise ValueError(f"Invalid topic number: valid topics numbers are 0 to {topic_count}")

    def _validate_topic_search(self, topic_num, doc_num):
        if doc_num > self.topic_sizes[topic_num]:
            raise ValueError(f"Invalid number of documents: topic {topic_num}"
                             f" only has {self.topic_sizes[topic_num]} documents")

    def _validate_doc_num(self, doc_num):
        self._less_than_zero(doc_num, "doc_num")
        document_count = len(self.documents) - 1
        if doc_num > document_count:
            raise ValueError(f"Invalid document number: valid document numbers are 0 to {document_count}")

    def _validate_keywords(self, keywords, keywords_neg):

        if not isinstance(keywords, list):
            raise ValueError("keywords must be a list of strings")

        if not isinstance(keywords_neg, list):
            raise ValueError("keywords_neg must be a list of strings")

        for word in keywords + keywords_neg:
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

    def get_topic_sizes(self):
        """
        Get topic sizes.

        The number of documents most similar to each topic. Topics are
        in increasing order of size.

        Returns
        -------
        topic_sizes: list
            The number of documents most similar to the topic.
        topic_nums:
            The unique index of every topic will be returned.
        """
        return self.topic_sizes.values.tolist(), self.topic_sizes.index.tolist()

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
        self._validate_topic_search(topic_num, num_docs)

        topic_document_indexes = np.where(self.doc_top == topic_num)[0]
        topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist[topic_document_indexes]))
        doc_nums = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
        doc_scores = self.doc_dist[doc_nums]

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
        sim_docs = self.model.docvecs.most_similar(positive=word_vecs,
                                                   negative=neg_word_vecs,
                                                   topn=num_docs)
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
        sim_words = self.model.wv.most_similar(positive=word_vecs,
                                               negative=neg_word_vecs,
                                               topn=num_words)
        words = [word[0] for word in sim_words]
        word_scores = [round(word[1], 4) for word in sim_words]

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

        combined_vector = np.zeros(300, dtype=np.float64)

        for word_vec in word_vecs:
            combined_vector += word_vec

        for word_vec in neg_word_vecs:
            combined_vector -= word_vec

        combined_vector /= (len(word_vecs) + len(neg_word_vecs))

        topic_ranks = [topic[0] for topic in cosine_similarity(self.topic_vectors, combined_vector.reshape(1, -1))]
        topic_nums = list(np.flip(np.argsort(topic_ranks)[-num_topics:]))

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
        sim_docs = self.model.docvecs.most_similar(positive=[message_vec],
                                                   topn=num_docs)
        doc_nums = [doc[0] for doc in sim_docs]
        doc_scores = [round(doc[1], 4) for doc in sim_docs]
        documents = list(itemgetter(*doc_nums)(self.documents))

        return documents, doc_scores, doc_nums

    def generate_topic_wordcloud(self, topic_num, background_color="black"):
        """
        Create a word cloud for a topic.

        A word cloud will be generated and displayed. The most semantically similar
        words to the topic will have the largest size, less similar words will be
        smaller. The size is determined using the cosine distance of the word vectors
        from the topic vector.

        Parameters
        ----------
        topic_num: int
            The topic number to search.

        background_color : str (Optional, default='white')
            Background color for the word cloud image. Suggested options are:
                * white
                * black

        Returns
        -------
        A matplotlib plot of the word cloud with the topic number will be displayed.

        """
        self._validate_topic_num(topic_num)

        word_score_dict = dict(zip(self.topic_words[topic_num], self.topic_word_scores[topic_num]))
        plt.figure(figsize=(16, 4),
                   dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(width=1600,
                      height=400,
                      background_color=background_color).generate_from_frequencies(word_score_dict))
        plt.title("Topic " + str(topic_num), loc='left', fontsize=25, pad=20)
