Top2Vec Model REST API
======================

Expose a trained and saved Top2Vec model with a REST API.

Docker Installation
------------
```bash
git clone https://github.com/ddangelov/Top2Vec.git
cd restful-top2vec
docker build --build-arg top2vec_model_path="/path_to_top2vec_model" -t restful-top2vec .
```

Run Container 
-------------
```bash
docker run -d --name model-name -p 80:80 restful-top2vec
```
