FROM tiangolo/uvicorn-gunicorn-fastapi:python3.7
COPY ./app /app
ARG top2vec_model_path
COPY $top2vec_model_path /app/top2vec_model
ENV model_path=/app/top2vec_model
COPY requirements.txt /app
RUN pip install -r requirements.txt
RUN pip uninstall top2vec -y
RUN pip install git+https://github.com/ddangelov/Top2Vec.git

# docker build --build-arg top2vec_model_path="/path/to/top2vec/model" -t restful-top2vec .
# docker run -d --name model-name -p 80:80 restful-top2vec