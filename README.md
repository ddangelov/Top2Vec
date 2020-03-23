Top2Vec
=======

Topic2Vector is a an algorithm for topic modeling. It automatically detects topics present in text
and generates jointly embedded topic, document and word vectors. Once you train the Top2Vec model 
you can:
* get number of detected topics
* get topics
* search topics by keywords
* search documents by topic
* find similar words
* find similar documents

The Algorithm
-------------

The assumption under the algorithm is that many documents that are semantically similar 
are indicative of an underlying topic. The first step is to create a joint embedding of 
document and word vectors. Once documents and words are embedded in a vector 
space the goal of the algorithm is to find dense clusters of documents, then identify which 
words attracted those documents together.

1. Create jointly embedded document and word vectors using Doc2Vec. 
2. Create lower dimensional embedding of document vectors using UMAP
3. Find dense areas of documents using HDBSCAN
4. For each dense area calculate centroid of document vectors in original dimension. (centroid = topic vector)
5. Find n-closest word vectors to the resulting topic vector


Installation
------------

The easy way to install Top2Vec is:

    pip install top2vec


Usage
-----

```python

from top2vec import Top2Vec

model = Top2Vec(documents)
```    

