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

