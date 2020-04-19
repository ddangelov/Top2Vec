[![](https://img.shields.io/pypi/v/top2vec.svg)](https://pypi.org/project/top2vec/)
[![](https://img.shields.io/pypi/l/top2vec.svg)](https://github.com/ddangelov/Top2Vec/blob/master/LICENSE)
[![](https://readthedocs.org/projects/top2vec/badge/?version=latest&token=0c691c6cc79b4906e35e8b7ede01e815baa05041d048945fa18e26810a3517d7)](https://top2vec.readthedocs.io/en/latest/?badge=latest)

Top2Vec Model REST API
======================

Expose a trained and saved Top2Vec model with a REST API.

Get Docker Image
------------
```bash
docker pull ddangelov/restful-top2vec
```

Run Container 
-------------

Docker Run Arguments:

  * ``model_path``: Path to a saved Top2Vec model.
  * ``model_name``: Name of Top2Vec model.
  
```bash
export model_path="/path_to_top2vec_model"
export model_name="model_name"

docker run -v $model_path:/app/top2vec_model -e model_name=$model_name -d --name model_name -p 80:80 ddangelov/restful-top2vec
```

Documentation
-------------

Go to <http://localhost:80/docs>

![](https://raw.githubusercontent.com/ddangelov/Top2Vec/master/images/restful-top2vec.png)
