Top2Vec Model REST API
======================

Expose a trained and saved Top2Vec model with a REST API.

Docker Installation
------------
```bash
git clone https://github.com/ddangelov/Top2Vec.git
cd Top2Vec/restful-top2vec
docker build -t restful-top2vec .
```

Run Container 
-------------

Docker Run Arguments:

  * ``model_path``: Path to a saved Top2Vec model.
  * ``model_name``: Name of Top2Vec model.
  
```bash
export model_path="/path_to_top2vec_model"
export model_name="model_name"
docker run -v $model_path:/app/top2vec_model -e model_name=$model_name  -d --name $model_name -p 80:80 restful-top2vec
```
Documentation
-------------

Go to <http://localhost:80/docs>

![](https://raw.githubusercontent.com/ddangelov/Top2Vec/master/images/restful-top2vec.png)
