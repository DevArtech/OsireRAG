FROM ghcr.io/abetlen/llama-cpp-python:v0.3.1

RUN apt-get update && apt-get install -y \
    build-essential \
    && pip install --upgrade pip pipenv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

COPY Pipfile Pipfile.lock /var/task/

COPY icon.png /var/task/

COPY app /var/task/app

RUN pipenv requirements > requirements.txt

RUN pip install -r requirements.txt --no-cache-dir

RUN python -m spacy download en_core_web_sm
RUN python -m nltk.downloader -d /usr/share/nltk_data punkt_tab stopwords wordnet

ENV PYTHONPATH=/var/task
ENV TOKENIZER_PATH=/var/task/app/models/tokenizer.pkl

CMD ["uvicorn", "app.main:app", "--port", "8080"]
