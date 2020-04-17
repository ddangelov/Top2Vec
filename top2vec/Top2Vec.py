# Author: Dimo Angelov
#
# License: BSD 3 clause
import logging
import numpy as np
import pandas as pd
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import strip_tags
import umap
import hdbscan
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from joblib import dump, load
from sklearn.cluster import dbscan

logger = logging.getLogger('top2vec')
logger.setLevel(logging.WARNING)
sh = logging.StreamHandler()
sh.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(sh)


def default_tokenizer(doc):
    """Tokenize documents for training and remove too long/short words"""
    return simple_preprocess(strip_tags(doc), deacc=True)


class Top2Vec:
    """
    Topic2Vector

    Creates jointly embedded topic, document and word vectors.


    Parameters
    ----------
    documents: List of str
        Input corpus, should be a list of strings.

    speed: string (Optional, default 'fast-learn')
        This parameter will determine how fast the model takes to train. The
        fast-learn option is the fastest and will generate the lowest quality
        vectors. The learn option will learn better quality vectors but take
        a longer time to train. The deep-learn option will learn the best quality
        vectors but will take significant time to train. The valid string speed
        options are:
            * fast-learn
            * learn
            * deep-learn

    document_ids: List of str, int (Optional)
        A unique value per document that will be used for referring to documents
        in search results. If ids are not given, the index of each document
        in the original corpus will become the id.

    keep_documents: bool (Optional, default True)
        If set to False documents will only be used for training and not saved
        as part of the model. This will reduce model size. When using search
        functions only document ids will be returned, not the actual documents.

    workers: int (Optional)
        The amount of worker threads to be used in training the model. Larger
        amount will lead to faster training.
    
    tokenizer: callable or None (default)
        Override the default tokenization method. If None then gensim.utils.simple_preprocess
        will be used.
    
    verbose: bool (Optional, default False)
        Whether to print status data during training.

    """

    def __init__(self, documents, speed="fast-learn", document_ids=None, keep_documents=True, workers=None,
                 tokenizer=None, verbose=False):

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.WARNING)

        # validate training inputs
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
        elif speed == "test-learn":
            hs = 0
            negative = 5
            epochs = 1
        else:
            raise ValueError("speed parameter needs to be one of: fast-learn, learn or deep-learn")

        if workers is None:
            pass
        elif isinstance(workers, int):
            pass
        else:
            raise ValueError("workers needs to be an int")

        if tokenizer is not None:
            self._tokenizer = tokenizer
        else:
            self._tokenizer = default_tokenizer

        # validate documents
        if not all((isinstance(doc, str) or isinstance(doc, np.str_)) for doc in documents):
            raise ValueError("Documents need to be a list of strings")
        if keep_documents:
            self.documents = np.array(documents, dtype="object")
        else:
            self.documents = None

        # validate document ids
        if document_ids is not None:

            if len(documents) != len(document_ids):
                raise ValueError("Document ids need to match number of documents")
            elif len(document_ids) != len(set(document_ids)):
                raise ValueError("Document ids need to be unique")

            if all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
                self.doc_id_type = np.str_
            elif all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
                self.doc_id_type = np.int_
            else:
                raise ValueError("Document ids need to be str or int")

            self.document_ids = np.array(document_ids)
            self.doc_id2index = dict(zip(document_ids, list(range(0, len(document_ids)))))
        else:
            self.document_ids = None
            self.doc_id2index = None
            self.doc_id_type = np.int_

        logger.info('Preprocessing documents for training')
        train_corpus = [TaggedDocument(self._tokenizer(doc), [i])
                        for i, doc in enumerate(documents)]

        # create documents and word embeddings with doc2vec
        logger.info('Creating joint document/word embedding')
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
        logger.info('Creating lower dimension embedding of documents')
        umap_model = umap.UMAP(n_neighbors=15,
                               n_components=5,
                               metric='cosine').fit(self.model.docvecs.vectors_docs)

        # find dense areas of document vectors
        logger.info('Finding dense areas of documents')
        cluster = hdbscan.HDBSCAN(min_cluster_size=15,
                                  metric='euclidean',
                                  cluster_selection_method='eom').fit(umap_model.embedding_)

        # calculate topic vectors from dense areas of documents
        logger.info('Finding topics')
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
        self.topic_vectors = np.vstack([self.model.docvecs.vectors_docs[np.where(cluster_labels == label)[0]]
                                       .mean(axis=0) for label in unique_labels])

    def _deduplicate_topics(self):
        core_samples, labels = dbscan(X=self.topic_vectors,
                                      eps=0.1,
                                      min_samples=2,
                                      metric="cosine")

        duplicate_clusters = set(labels)

        if len(duplicate_clusters) > 1 or -1 not in duplicate_clusters:

            # unique topics
            unique_topics = self.topic_vectors[np.where(labels == -1)[0]]

            if -1 in duplicate_clusters:
                duplicate_clusters.remove(-1)

            # merge duplicate topics
            for unique_label in duplicate_clusters:
                unique_topics = np.vstack(
                    [unique_topics, self.topic_vectors[np.where(labels == unique_label)[0]]
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
        self.topic_words = np.array(self.topic_words)
        self.topic_word_scores = np.array(self.topic_word_scores)

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
        document_count = self.model.docvecs.count
        if num_docs > self.model.docvecs.count:
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

    def _validate_topic_search(self, topic_num, num_docs):
        if num_docs > self.topic_sizes[topic_num]:
            raise ValueError(f"Invalid number of documents: topic {topic_num}"
                             f" only has {self.topic_sizes[topic_num]} documents")

    def _validate_doc_ids(self, doc_ids, doc_ids_neg):

        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list of string or int")

        if not isinstance(doc_ids_neg, list):
            raise ValueError("doc_ids_neg must be a list of string or int")

        doc_ids_all = doc_ids + doc_ids_neg
        for doc_id in doc_ids_all:
            if self.document_ids is not None:
                if doc_id not in self.document_ids:
                    raise ValueError(f"{doc_id} is not a valid document id")
            elif doc_id < 0 or doc_id > self.model.docvecs.count - 1:
                raise ValueError(f"{doc_id} is not a valid document id")

    def _validate_keywords(self, keywords, keywords_neg):

        if not (isinstance(keywords, list) or isinstance(keywords, np.ndarray)):
            raise ValueError("keywords must be a list of strings")

        if not (isinstance(keywords_neg, list) or isinstance(keywords_neg, np.ndarray)):
            raise ValueError("keywords_neg must be a list of strings")

        keywords_lower = [keyword.lower() for keyword in keywords]
        keywords_neg_lower = [keyword.lower() for keyword in keywords_neg]

        for word in keywords_lower + keywords_neg_lower:
            if word not in self.model.wv.vocab:
                raise ValueError(f"'{word}' has not been learned by the model so it cannot be searched")

        return keywords_lower, keywords_neg_lower

    def _get_document_ids(self, doc_index):
        if self.document_ids is None:
            return doc_index
        else:
            return self.document_ids[doc_index]

    def _get_document_indexes(self, doc_ids):
        if self.document_ids is None:
            return doc_ids
        else:
            return [self.doc_id2index[doc_id] for doc_id in doc_ids]

    def _get_word_vectors(self, keywords):
        return [self.model[word] for word in keywords]

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
        topic_sizes: array of int, shape(num_topics)
            The number of documents most similar to the topic.
        topic_nums: array of int, shape(num_topics)
            The unique number of every topic will be returned.
        """
        return np.array(self.topic_sizes.values), np.array(self.topic_sizes.index)

    def get_topics(self, num_topics):
        """
        Get number of specified topics, ordered by decreasing size.

        Each topic will consist of the top 50 semantically similar words
        to the topic. These are the 50 words closest to topic vector
        along with cosine similarity of each word from vector. The
        higher the score the more relevant the word is to the topic.

        Parameters
        ----------
        num_topics: int, shape(num_topics)
            Number of topics to return.

        Returns
        -------
        topics_words: array of shape(num_topics, 50)
            For each topic the top 50 words are returned, in order
            of semantic similarity to topic.
            Example:
                [['data', 'deep', 'learning' ... 'artificial'],          <Topic 0>
                 ['environment', 'warming', 'climate ... 'temperature']  <Topic 1>
                 ...]

        word_scores: array of shape(num_topics, 50)
            For each topic the cosine similarity scores of the
            top 50 words to the topic are returned.
            Example:
                [[0.7132, 0.6473, 0.5700 ... 0.3455],  <Topic 0>
                 [0.7818', 0.7671, 0.7603 ... 0.6769]  <Topic 1>
                 ...]

        topic_nums: array of int, shape(num_topics)
            The unique number of every topic will be returned.
        """

        self._validate_num_topics(num_topics)

        return self.topic_words[0:num_topics], self.topic_word_scores[0:num_topics], np.array(range(0, num_topics))

    def search_documents_by_topic(self, topic_num, num_docs, return_documents=True):
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

        return_documents: bool (Optional default True)
            Determines if the documents will be returned. If they were not saved
            in the model they will also not be returned.

        Returns
        -------
        documents: (Optional) array of str, shape(num_docs)
            The documents in a list, the most similar are first.

            Will only be returned if the documents were saved and if return_documents
            is set to True.

        doc_scores: array of float, shape(num_docs)
            Semantic similarity of document to topic. The cosine similarity of the
            document and topic vector.

        doc_ids: array of int, shape(num_docs)
            Unique ids of documents. If ids were not given, the index of document
            in the original corpus will be returned.
        """
        self._validate_num_docs(num_docs)
        self._validate_topic_num(topic_num)
        self._validate_topic_search(topic_num, num_docs)

        topic_document_indexes = np.where(self.doc_top == topic_num)[0]
        topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist[topic_document_indexes]))
        doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
        doc_scores = self.doc_dist[doc_indexes]
        doc_ids = self._get_document_ids(doc_indexes)

        if self.documents is not None and return_documents:
            documents = self.documents[doc_indexes]
            return documents, doc_scores, doc_ids
        else:
            return doc_scores, doc_ids

    def search_documents_by_keywords(self, keywords, num_docs, keywords_neg=[], return_documents=True):
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
        keywords: List of str
            List of positive keywords being used for search of semantically similar
            documents.

        keywords_neg: List of str (Optional)
            List of negative keywords being used for search of semantically dissimilar
            documents.

        num_docs: int
            Number of documents to return.

        return_documents: bool (Optional default True)
            Determines if the documents will be returned. If they were not saved
            in the model they will also not be returned.

        Returns
        -------
        documents: (Optional) array of str, shape(num_docs)
            The documents in a list, the most similar are first.

            Will only be returned if the documents were saved and if return_documents
            is set to True.

        doc_scores: array of float, shape(num_docs)
            Semantic similarity of document to keywords. The cosine similarity of the
            document and average of keyword vectors.

        doc_ids: array of int, shape(num_docs)
            Unique ids of documents. If ids were not given, the index of document
            in the original corpus will be returned.
        """
        self._validate_num_docs(num_docs)
        keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)

        word_vecs = self._get_word_vectors(keywords)
        neg_word_vecs = self._get_word_vectors(keywords_neg)
        sim_docs = self.model.docvecs.most_similar(positive=word_vecs,
                                                   negative=neg_word_vecs,
                                                   topn=num_docs)
        doc_indexes = [doc[0] for doc in sim_docs]
        doc_scores = np.array([round(doc[1], 4) for doc in sim_docs])
        doc_ids = self._get_document_ids(doc_indexes)

        if self.documents is not None and return_documents:
            documents = self.documents[doc_indexes]
            return documents, doc_scores, doc_ids
        else:
            return doc_scores, doc_ids

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
        keywords: List of str
            List of positive keywords being used for search of semantically similar
            words.

        keywords_neg: List of str
            List of negative keywords being used for search of semantically dissimilar
            words.

        num_words: int
            Number of words to return.


        Returns
        -------
        words: array of str, shape(num_words)
            The words in a list, the most similar are first.

        word_scores: array of float, shape(num_words)
            Semantic similarity of word to keywords. The cosine similarity of the
            word and average of keyword vectors.
        """
        keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)
        sim_words = self.model.wv.most_similar(positive=keywords,
                                               negative=keywords_neg,
                                               topn=num_words)
        words = np.array([word[0] for word in sim_words])
        word_scores = np.array([round(word[1], 4) for word in sim_words])

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
        keywords: List of str
            List of positive keywords being used for search of semantically similar
            documents.

        keywords_neg: (Optional) List of str
            List of negative keywords being used for search of semantically dissimilar
            documents.

        num_topics: int
            Number of documents to return.

        Returns
        -------
        topics_words: array of shape (num_topics, 50)
            For each topic the top 50 words are returned, in order of semantic
            similarity to topic.
            Example:
                [['data', 'deep', 'learning' ... 'artificial'],             <Topic 0>
                 ['environment', 'warming', 'climate ... 'temperature']     <Topic 1>
                 ...]

        word_scores: array of shape (num_topics, 50)
            For each topic the cosine similarity scores of the top 50 words
            to the topic are returned.
            Example:
                [[0.7132, 0.6473, 0.5700 ... 0.3455],     <Topic 0>
                 [0.7818', 0.7671, 0.7603 ... 0.6769]     <Topic 1>
                 ...]

        topic_scores: array of float, shape(num_topics)
            For each topic the cosine similarity to the search keywords will be
            returned.

        topic_nums: array of int, shape(num_topics)
            The unique number of every topic will be returned.
        """
        self._validate_num_topics(num_topics)
        keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)

        word_vecs = self._get_word_vectors(keywords)
        neg_word_vecs = self._get_word_vectors(keywords_neg)

        combined_vector = np.zeros(300, dtype=np.float64)

        for word_vec in word_vecs:
            combined_vector += word_vec

        for word_vec in neg_word_vecs:
            combined_vector -= word_vec

        combined_vector /= (len(word_vecs) + len(neg_word_vecs))

        topic_ranks = [topic[0] for topic in cosine_similarity(self.topic_vectors, combined_vector.reshape(1, -1))]
        topic_nums = np.flip(np.argsort(topic_ranks)[-num_topics:])

        topic_words = [self.topic_words[topic] for topic in topic_nums]
        word_scores = [self.topic_word_scores[topic] for topic in topic_nums]
        topic_scores = np.array([round(topic_ranks[topic], 4) for topic in topic_nums])

        return topic_words, word_scores, topic_scores, topic_nums

    def search_documents_by_documents(self, doc_ids, num_docs, doc_ids_neg=[], return_documents=True):
        """
        Semantic similarity search of documents.

        The most semantically similar documents to the semantic combination of
        document ids provided will be returned. If negative document ids are provided,
        the documents will be semantically dissimilar to those document ids. Documents
        will be ordered by decreasing similarity. This method finds the closest document
        vectors to the provided documents averaged.

        Parameters
        ----------
        doc_ids: List of int, str
            Unique ids of document. If ids were not given, the index of document
            in the original corpus.

        doc_ids_neg: (Optional) List of int, str
            Unique ids of document. If ids were not given, the index of document
            in the original corpus.

        num_docs: int
            Number of documents to return.

        return_documents: bool (Optional default True)
            Determines if the documents will be returned. If they were not saved
            in the model they will also not be returned.

        Returns
        -------
        documents: (Optional) array of str, shape(num_docs)
            The documents in a list, the most similar are first.

            Will only be returned if the documents were saved and if return_documents
            is set to True.

        doc_scores: array of float, shape(num_docs)
            Semantic similarity of document to keywords. The cosine similarity of the
            document and average of keyword vectors.

        doc_ids: array of int, shape(num_docs)
            Unique ids of documents. If ids were not given, the index of document
            in the original corpus will be returned.
        """
        self._validate_num_docs(num_docs)
        self._validate_doc_ids(doc_ids, doc_ids_neg)

        doc_indexes = self._get_document_indexes(doc_ids)
        doc_indexes_neg = self._get_document_indexes(doc_ids_neg)
        sim_docs = self.model.docvecs.most_similar(positive=doc_indexes,
                                                   negative=doc_indexes_neg,
                                                   topn=num_docs)
        doc_indexes = [doc[0] for doc in sim_docs]
        doc_scores = np.array([round(doc[1], 4) for doc in sim_docs])
        doc_ids = self._get_document_ids(doc_indexes)

        if self.documents is not None and return_documents:
            documents = self.documents[doc_indexes]
            return documents, doc_scores, doc_ids
        else:
            return doc_scores, doc_ids

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
