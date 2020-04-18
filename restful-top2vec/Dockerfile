FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./app /app
ARG model_path
ARG model_name="Top2Vec API"
COPY $model_path /app/top2vec_model
ENV model_name=$model_name
ENV model_path=/app/top2vec_model
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip uninstall top2vec -y
RUN pip install git+https://github.com/ddangelov/Top2Vec.git