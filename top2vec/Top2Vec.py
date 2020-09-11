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
import tempfile

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
    Top2Vec

    Creates jointly embedded topic, document and word vectors.


    Parameters
    ----------
    documents: List of str
        Input corpus, should be a list of strings.

    min_count: int (Optional, default 50)
        Ignores all words with total frequency lower than this. For smaller
        corpora a smaller min_count will be necessary.

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

    use_corpus_file: bool (Optional, default False)
        Setting use_corpus_file to True can sometimes provide speedup for large
        datasets when multiple worker threads are available. Documents are still
        passed to the model as a list of str, the model will create a temporary
        corpus file for training.

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
    
    tokenizer: callable (Optional, default None)
        Override the default tokenization method. If None then gensim.utils.simple_preprocess
        will be used.
    
    verbose: bool (Optional, default False)
        Whether to print status data during training.

    """

    def __init__(self, documents, min_count=50, speed="fast-learn", use_corpus_file=False, document_ids=None,
                 keep_documents=True, workers=None, tokenizer=None, verbose=False):

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

        logger.info('Pre-processing documents for training')
        if use_corpus_file:
            processed = [self._tokenizer(doc) for doc in documents]
            lines = [' '.join(line) + "\n" for line in processed]
            temp = tempfile.NamedTemporaryFile(mode='w+t')
            temp.writelines(lines)
        else:
            train_corpus = [TaggedDocument(self._tokenizer(doc), [i])
                            for i, doc in enumerate(documents)]

        # create documents and word embeddings with doc2vec
        logger.info('Creating joint document/word embedding')
        if use_corpus_file:
            if workers is None:
                self.model = Doc2Vec(corpus_file=temp.name,
                                     vector_size=300,
                                     min_count=min_count,
                                     window=15,
                                     sample=1e-5,
                                     negative=negative,
                                     hs=hs,
                                     epochs=epochs,
                                     dm=0,
                                     dbow_words=1)
            else:
                self.model = Doc2Vec(corpus_file=temp.name,
                                     vector_size=300,
                                     min_count=min_count,
                                     window=15,
                                     sample=1e-5,
                                     negative=negative,
                                     hs=hs,
                                     workers=workers,
                                     epochs=epochs,
                                     dm=0,
                                     dbow_words=1)

            temp.close()
        else:
            if workers is None:
                self.model = Doc2Vec(documents=train_corpus,
                                     vector_size=300,
                                     min_count=min_count,
                                     window=15,
                                     sample=1e-5,
                                     negative=negative,
                                     hs=hs,
                                     epochs=epochs,
                                     dm=0,
                                     dbow_words=1)
            else:
                self.model = Doc2Vec(documents=train_corpus,
                                     vector_size=300,
                                     min_count=min_count,
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
        self.topic_vectors, self.doc_top, self.doc_dist, self.topic_sizes = self._calculate_topic_sizes(
            self.topic_vectors)

        # find topic words and scores
        self.topic_words, self.topic_word_scores = self._find_topic_words_scores(topic_vectors=self.topic_vectors)

        # initialize variables for hierarchical topic reduction
        self.topic_vectors_reduced = None
        self.doc_top_reduced = None
        self.doc_dist_reduced = None
        self.topic_sizes_reduced = None
        self.topic_words_reduced = None
        self.topic_word_scores_reduced = None
        self.hierarchy = None

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

    def _calculate_topic_sizes(self, topic_vectors, hierarchy=None):
        # find nearest topic of each document
        doc_top, doc_dist = self._calculate_documents_topic(topic_vectors=topic_vectors,
                                                            document_vectors=self.model.docvecs.vectors_docs)
        topic_sizes = pd.Series(doc_top).value_counts()

        return self._reorder_topics(topic_vectors, topic_sizes, doc_top, doc_dist, hierarchy)

    @staticmethod
    def _reorder_topics(topic_vectors, topic_sizes, doc_top, doc_dist, hierarchy=None):
        topic_vectors = topic_vectors[topic_sizes.index]
        old2new = dict(zip(topic_sizes.index, range(topic_sizes.shape[0])))
        doc_top = np.array([old2new[i] for i in doc_top])

        if hierarchy is None:
            topic_sizes.reset_index(drop=True, inplace=True)
            return topic_vectors, doc_top, doc_dist, topic_sizes

        else:
            hierarchy = [hierarchy[i] for i in topic_sizes.index]
            topic_sizes.reset_index(drop=True, inplace=True)
            return topic_vectors, doc_top, doc_dist, topic_sizes, hierarchy

    @staticmethod
    def _calculate_documents_topic(topic_vectors, document_vectors, dist=True):
        batch_size = 10000
        doc_top = []
        if dist:
            doc_dist = []

        if document_vectors.shape[0] > batch_size:
            current = 0
            batches = int(document_vectors.shape[0] / batch_size)
            extra = document_vectors.shape[0] % batch_size

            for ind in range(0, batches):
                res = cosine_similarity(document_vectors[current:current + batch_size], topic_vectors)
                doc_top.extend(np.argmax(res, axis=1))
                if dist:
                    doc_dist.extend(np.max(res, axis=1))
                current += batch_size

            if extra > 0:
                res = cosine_similarity(document_vectors[current:current + extra], topic_vectors)
                doc_top.extend(np.argmax(res, axis=1))
                if dist:
                    doc_dist.extend(np.max(res, axis=1))
            if dist:
                doc_dist = np.array(doc_dist)
        else:
            res = cosine_similarity(document_vectors, topic_vectors)
            doc_top = np.argmax(res, axis=1)
            if dist:
                doc_dist = np.max(res, axis=1)

        if dist:
            return doc_top, doc_dist
        else:
            return doc_top

    def _find_topic_words_scores(self, topic_vectors):
        topic_words = []
        topic_word_scores = []

        for topic_vector in topic_vectors:
            sim_words = self.model.wv.most_similar(positive=[topic_vector], topn=50)
            topic_words.append([word[0] for word in sim_words])
            topic_word_scores.append([round(word[1], 4) for word in sim_words])

        topic_words = np.array(topic_words)
        topic_word_scores = np.array(topic_word_scores)

        return topic_words, topic_word_scores

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

    def _validate_hierarchical_reduction(self):
        if self.hierarchy is None:
            raise ValueError("Hierarchical topic reduction has not been performed.")

    def _validate_hierarchical_reduction_num_topics(self, num_topics):
        current_num_topics = len(self.topic_vectors)
        if num_topics >= current_num_topics:
            raise ValueError(f"Number of topics must be less than {current_num_topics}.")

    def _validate_num_docs(self, num_docs):
        self._less_than_zero(num_docs, "num_docs")
        document_count = self.model.docvecs.count
        if num_docs > self.model.docvecs.count:
            raise ValueError(f"num_docs cannot exceed the number of documents: {document_count}.")

    def _validate_num_topics(self, num_topics, reduced):
        self._less_than_zero(num_topics, "num_topics")
        if reduced:
            topic_count = len(self.topic_vectors_reduced)
            if num_topics > topic_count:
                raise ValueError(f"num_topics cannot exceed the number of reduced topics: {topic_count}.")
        else:
            topic_count = len(self.topic_vectors)
            if num_topics > topic_count:
                raise ValueError(f"num_topics cannot exceed the number of topics: {topic_count}.")

    def _validate_topic_num(self, topic_num, reduced):
        self._less_than_zero(topic_num, "topic_num")

        if reduced:
            topic_count = len(self.topic_vectors_reduced) - 1
            if topic_num > topic_count:
                raise ValueError(f"Invalid topic number: valid reduced topics numbers are 0 to {topic_count}.")
        else:
            topic_count = len(self.topic_vectors) - 1
            if topic_num > topic_count:
                raise ValueError(f"Invalid topic number: valid original topics numbers are 0 to {topic_count}.")

    def _validate_topic_search(self, topic_num, num_docs, reduced):
        self._less_than_zero(num_docs, "num_docs")
        if reduced:
            if num_docs > self.topic_sizes_reduced[topic_num]:
                raise ValueError(f"Invalid number of documents: reduced topic {topic_num}"
                                 f" only has {self.topic_sizes_reduced[topic_num]} documents.")
        else:
            if num_docs > self.topic_sizes[topic_num]:
                raise ValueError(f"Invalid number of documents: original topic {topic_num}"
                                 f" only has {self.topic_sizes[topic_num]} documents.")

    def _validate_doc_ids(self, doc_ids, doc_ids_neg):

        if not isinstance(doc_ids, list):
            raise ValueError("doc_ids must be a list of string or int.")
        if not isinstance(doc_ids_neg, list):
            raise ValueError("doc_ids_neg must be a list of string or int.")

        doc_ids_all = doc_ids + doc_ids_neg
        for doc_id in doc_ids_all:
            if self.document_ids is not None:
                if doc_id not in self.document_ids:
                    raise ValueError(f"{doc_id} is not a valid document id.")
            elif doc_id < 0 or doc_id > self.model.docvecs.count - 1:
                raise ValueError(f"{doc_id} is not a valid document id.")

    def _validate_keywords(self, keywords, keywords_neg):
        if not (isinstance(keywords, list) or isinstance(keywords, np.ndarray)):
            raise ValueError("keywords must be a list of strings.")

        if not (isinstance(keywords_neg, list) or isinstance(keywords_neg, np.ndarray)):
            raise ValueError("keywords_neg must be a list of strings.")

        keywords_lower = [keyword.lower() for keyword in keywords]
        keywords_neg_lower = [keyword.lower() for keyword in keywords_neg]

        for word in keywords_lower + keywords_neg_lower:
            if word not in self.model.wv.vocab:
                raise ValueError(f"'{word}' has not been learned by the model so it cannot be searched.")

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

    def _validate_document_ids_add_doc(self, documents, document_ids):
        if document_ids is None:
            raise ValueError("Document ids need to be provided.")
        if len(documents) != len(document_ids):
            raise ValueError("Document ids need to match number of documents.")
        elif len(document_ids) != len(set(document_ids)):
            raise ValueError("Document ids need to be unique.")

        if all((isinstance(doc_id, str) or isinstance(doc_id, np.str_)) for doc_id in document_ids):
            if self.doc_id_type == np.int_:
                raise ValueError("Document ids need to be of type int.")
        elif all((isinstance(doc_id, int) or isinstance(doc_id, np.int_)) for doc_id in document_ids):
            if self.doc_id_type == np.str_:
                raise ValueError("Document ids need to be of type str.")

        if len(set(document_ids).intersection(self.document_ids)) > 0:
            raise ValueError("Some document ids already exist in model.")

    @staticmethod
    def _validate_documents(documents):
        if not all((isinstance(doc, str) or isinstance(doc, np.str_)) for doc in documents):
            raise ValueError("Documents need to be a list of strings.")

    def _assign_documents_to_topic(self, document_vectors, topic_vectors, topic_sizes, doc_top, doc_dist,
                                   hierarchy=None):

        doc_top_new, doc_dist_new = self._calculate_documents_topic(topic_vectors, document_vectors, dist=True)
        doc_top = np.append(doc_top, doc_top_new)
        doc_dist = np.append(doc_dist, doc_dist_new)

        topic_sizes_new = pd.Series(doc_top_new).value_counts()
        for top in topic_sizes_new.index.tolist():
            topic_sizes[top] += topic_sizes_new[top]
        topic_sizes.sort_values(ascending=False, inplace=True)

        if hierarchy is None:
            return self._reorder_topics(topic_vectors, topic_sizes, doc_top, doc_dist)
        else:
            return self._reorder_topics(topic_vectors, topic_sizes, doc_top, doc_dist, hierarchy)

    def add_documents(self, documents, document_ids=None):
        """
        Update the model with new documents.

        The documents will be added to the current model without changing
        existing document, word and topic vectors. Topic sizes will be updated.

        If adding a large quantity of documents relative to the current model size,
        or documents containing a largely new vocabulary, a new model should be
        trained for best results.

        Parameters
        ----------
        documents: List of str

        document_ids: List of str, int (Optional)

            Only required if document_ids were given to the original model.

            A unique value per document that will be used for referring to documents
            in search results. If ids are not given, the index of each document
            in the original corpus will become the id.


        """

        # add documents
        self._validate_documents(documents)
        if self.documents is not None:
            self.documents = np.append(self.documents, documents)

        # add document ids
        if self.document_ids is not None:
            self._validate_document_ids_add_doc(documents, document_ids)
            doc_ids_len = len(self.document_ids)
            self.document_ids = np.append(self.document_ids, document_ids)
            self.doc_id2index.update(dict(zip(document_ids, list(range(doc_ids_len, doc_ids_len + len(document_ids))))))

        # get document vectors
        docs_processed = [self._tokenizer(doc) for doc in documents]
        document_vectors = np.vstack([self.model.infer_vector(doc_words=doc, alpha=0.025, min_alpha=0.01, epochs=100)
                                      for doc in docs_processed])

        # add documents do model
        num_docs = len(documents)
        self.model.docvecs.vectors_docs = np.vstack([self.model.docvecs.vectors_docs, document_vectors])
        self.model.docvecs.count += num_docs
        self.model.docvecs.max_rawint += num_docs
        self.model.docvecs.vectors_docs_norm = None
        self.model.docvecs.init_sims()

        # update topics
        self.topic_vectors, self.doc_top, self.doc_dist, self.topic_sizes = self._assign_documents_to_topic(
            document_vectors,
            self.topic_vectors,
            self.topic_sizes,
            self.doc_top,
            self.doc_dist)

        if self.hierarchy is not None:
            self.topic_vectors_reduced, self.doc_top_reduced,\
                self.doc_dist_reduced, self.topic_sizes_reduced, self.hierarchy = self._assign_documents_to_topic(
                    document_vectors,
                    self.topic_vectors_reduced,
                    self.topic_sizes_reduced,
                    self.doc_top_reduced,
                    self.doc_dist_reduced,
                    self.hierarchy)

    def get_num_topics(self, reduced=False):
        """
        Get number of topics.

        This is the number of topics Top2Vec has found in the data by default.
        If reduced is True, the number of reduced topics is returned.

        Parameters
        ----------
        reduced: bool (Optional, default False)
            The number of original topics will be returned by default. If True will
            return the number of reduced topics, if hierarchical topic reduction
            has been performed.

        Returns
        -------
        num_topics: int
        """

        if reduced:
            self._validate_hierarchical_reduction()
            return len(self.topic_vectors_reduced)
        else:
            return len(self.topic_vectors)

    def get_topic_sizes(self, reduced=False):
        """
        Get topic sizes.

        The number of documents most similar to each topic. Topics are
        in increasing order of size.

        The sizes of the original topics is returned unless reduced=True,
        in which case the sizes of the reduced topics will be returned.

        Parameters
        ----------
        reduced: bool (Optional, default False)
            Original topic sizes are returned by default. If True the
            reduced topic sizes will be returned.

        Returns
        -------
        topic_sizes: array of int, shape(num_topics)
            The number of documents most similar to the topic.
        topic_nums: array of int, shape(num_topics)
            The unique number of every topic will be returned.
        """
        if reduced:
            self._validate_hierarchical_reduction()
            return np.array(self.topic_sizes_reduced.values), np.array(self.topic_sizes_reduced.index)
        else:
            return np.array(self.topic_sizes.values), np.array(self.topic_sizes.index)

    def get_topics(self, num_topics=None, reduced=False):
        """
        Get topics, ordered by decreasing size. All topics are returned
        if num_topics is not specified.

        The original topics found are returned unless reduced=True,
        in which case reduced topics will be returned.

        Each topic will consist of the top 50 semantically similar words
        to the topic. These are the 50 words closest to topic vector
        along with cosine similarity of each word from vector. The
        higher the score the more relevant the word is to the topic.

        Parameters
        ----------
        num_topics: int, (Optional)
            Number of topics to return.

        reduced: bool (Optional, default False)
            Original topics are returned by default. If True the
            reduced topics will be returned.

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
        if reduced:
            self._validate_hierarchical_reduction()

            if num_topics is None:
                num_topics = len(self.topic_vectors_reduced)
            else:
                self._validate_num_topics(num_topics, reduced)

            return self.topic_words_reduced[0:num_topics], self.topic_word_scores_reduced[0:num_topics], np.array(
                range(0, num_topics))
        else:

            if num_topics is None:
                num_topics = len(self.topic_vectors)
            else:
                self._validate_num_topics(num_topics, reduced)

            return self.topic_words[0:num_topics], self.topic_word_scores[0:num_topics], np.array(range(0, num_topics))

    def get_topic_hierarchy(self):
        """
        Get the hierarchy of reduced topics. The mapping of each original topic to the reduced
        topics is returned.

        Hierarchical topic reduction must be performed before calling this method.

        Returns
        -------
        hierarchy: list of ints
            Each index of the hierarchy corresponds to the topic number of a reduced topic.
            For each reduced topic the topic numbers of the original topics that were merged
            to create it are listed.

            Example:
            [[3]  <Reduced Topic 0> contains original Topic 3
            [2,4] <Reduced Topic 1> contains original Topics 2 and 4
            [0,1] <Reduced Topic 3> contains original Topics 0 and 1
            ...]
        """

        self._validate_hierarchical_reduction()

        return self.hierarchy

    def hierarchical_topic_reduction(self, num_topics):
        """
        Reduce the number of topics discovered by Top2Vec.

        The most representative topics of the corpus will be found, by iteratively merging
        each smallest topic to the most similar topic until num_topics is reached.

        Parameters
        ----------
        num_topics: int
            The number of topics to reduce to.

        Returns
        -------
        hierarchy: list of ints
            Each index of hierarchy corresponds to the reduced topics, for each reduced
            topic the indexes of the original topics that were merged to create it are
            listed.

            Example:
            [[3]  <Reduced Topic 0> contains original Topic 3
            [2,4] <Reduced Topic 1> contains original Topics 2 and 4
            [0,1] <Reduced Topic 3> contains original Topics 0 and 1
            ...]
        """
        self._validate_hierarchical_reduction_num_topics(num_topics)

        num_topics_current = self.topic_vectors.shape[0]
        top_vecs = self.topic_vectors
        top_sizes = [self.topic_sizes[i] for i in range(0, len(self.topic_sizes))]
        hierarchy = [[i] for i in range(self.topic_vectors.shape[0])]

        count = 0
        interval = max(int(self.model.docvecs.count / 50000), 1)

        while num_topics_current > num_topics:

            # find smallest and most similar topics
            sizes = pd.Series(top_sizes).sort_values(ascending=False)
            smallest = sizes.sort_values(ascending=True).index[0]
            res = cosine_similarity(top_vecs[smallest].reshape(1, -1), top_vecs)
            most_sim = np.flip(np.argsort(res[0]))[1]

            # calculate combined topic vector
            top_vec_smallest = top_vecs[smallest]
            smallest_size = top_sizes[smallest]

            top_vec_most_sim = top_vecs[most_sim]
            most_sim_size = top_sizes[most_sim]

            combined_vec = ((top_vec_smallest * smallest_size) + (top_vec_most_sim * most_sim_size)) / (
                    smallest_size + most_sim_size)

            # update topic vectors
            ix_keep = list(range(len(top_vecs)))
            ix_keep.remove(smallest)
            ix_keep.remove(most_sim)
            top_vecs = top_vecs[ix_keep]
            top_vecs = np.vstack([top_vecs, combined_vec])
            num_topics_current = top_vecs.shape[0]

            # update topics sizes
            if count % interval == 0:
                doc_top = self._calculate_documents_topic(topic_vectors=top_vecs,
                                                          document_vectors=self.model.docvecs.vectors_docs,
                                                          dist=False)
                topic_sizes = pd.Series(doc_top).value_counts()
                top_sizes = [topic_sizes[i] for i in range(0, len(topic_sizes))]

            else:
                smallest_size = top_sizes.pop(smallest)
                if most_sim < smallest:
                    most_sim_size = top_sizes.pop(most_sim)
                else:
                    most_sim_size = top_sizes.pop(most_sim - 1)
                combined_size = smallest_size + most_sim_size
                top_sizes.append(combined_size)

            count += 1

            # update topic hierarchy
            smallest_inds = hierarchy.pop(smallest)
            if most_sim < smallest:
                most_sim_inds = hierarchy.pop(most_sim)
            else:
                most_sim_inds = hierarchy.pop(most_sim - 1)

            combined_inds = smallest_inds + most_sim_inds
            hierarchy.append(combined_inds)

        # re-calculate topic vectors from clusters
        doc_top = self._calculate_documents_topic(topic_vectors=top_vecs,
                                                  document_vectors=self.model.docvecs.vectors_docs,
                                                  dist=False)
        top_vecs = np.vstack([self.model.docvecs.vectors_docs[np.where(doc_top == label)[0]].mean(axis=0)
                              for label in set(doc_top)])
        self.topic_vectors_reduced, self.doc_top_reduced, self.doc_dist_reduced, self.topic_sizes_reduced, \
        self.hierarchy = self._calculate_topic_sizes(topic_vectors=top_vecs,
                                                     hierarchy=hierarchy)
        self.topic_words_reduced, self.topic_word_scores_reduced = self._find_topic_words_scores(
            topic_vectors=self.topic_vectors_reduced)

        return self.hierarchy

    def search_documents_by_topic(self, topic_num, num_docs, return_documents=True, reduced=False):
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

        reduced: bool (Optional, default False)
            Original topics are used to search by default. If True the
            reduced topics will be used.

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

        if reduced:
            self._validate_hierarchical_reduction()
            self._validate_topic_num(topic_num, reduced)
            self._validate_topic_search(topic_num, num_docs, reduced)

            topic_document_indexes = np.where(self.doc_top_reduced == topic_num)[0]
            topic_document_indexes_ordered = np.flip(np.argsort(self.doc_dist_reduced[topic_document_indexes]))
            doc_indexes = topic_document_indexes[topic_document_indexes_ordered][0:num_docs]
            doc_scores = self.doc_dist_reduced[doc_indexes]
            doc_ids = self._get_document_ids(doc_indexes)

        else:

            self._validate_topic_num(topic_num, reduced)
            self._validate_topic_search(topic_num, num_docs, reduced)

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

    def search_topics(self, keywords, num_topics, keywords_neg=[], reduced=False):
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

        reduced: bool (Optional, default False)
            Original topics are searched by default. If True the
            reduced topics will be searched.

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
        self._validate_num_topics(num_topics, reduced)
        keywords, keywords_neg = self._validate_keywords(keywords, keywords_neg)

        word_vecs = self._get_word_vectors(keywords)
        neg_word_vecs = self._get_word_vectors(keywords_neg)

        combined_vector = np.zeros(300, dtype=np.float64)

        for word_vec in word_vecs:
            combined_vector += word_vec

        for word_vec in neg_word_vecs:
            combined_vector -= word_vec

        combined_vector /= (len(word_vecs) + len(neg_word_vecs))

        if reduced:
            self._validate_hierarchical_reduction()

            topic_ranks = [topic[0] for topic in
                           cosine_similarity(self.topic_vectors_reduced, combined_vector.reshape(1, -1))]
            topic_nums = np.flip(np.argsort(topic_ranks)[-num_topics:])
            topic_words = [self.topic_words_reduced[topic] for topic in topic_nums]
            word_scores = [self.topic_word_scores_reduced[topic] for topic in topic_nums]
            topic_scores = np.array([round(topic_ranks[topic], 4) for topic in topic_nums])
        else:
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

    def generate_topic_wordcloud(self, topic_num, background_color="black", reduced=False):
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

        reduced: bool (Optional, default False)
            Original topics are used by default. If True the
            reduced topics will be used.

        Returns
        -------
        A matplotlib plot of the word cloud with the topic number will be displayed.

        """

        if reduced:
            self._validate_hierarchical_reduction()
            self._validate_topic_num(topic_num, reduced)
            word_score_dict = dict(zip(self.topic_words_reduced[topic_num], self.topic_word_scores_reduced[topic_num]))
        else:
            self._validate_topic_num(topic_num, reduced)
            word_score_dict = dict(zip(self.topic_words[topic_num], self.topic_word_scores[topic_num]))

        plt.figure(figsize=(16, 4),
                   dpi=200)
        plt.axis("off")
        plt.imshow(
            WordCloud(width=1600,
                      height=400,
                      background_color=background_color).generate_from_frequencies(word_score_dict))
        plt.title("Topic " + str(topic_num), loc='left', fontsize=25, pad=20)
