# syntax=docker/dockerfile:1.4

FROM ghcr.io/abetlen/llama-cpp-python:v0.3.1 as builder

# Install build dependencies
RUN apt-get update && apt-get install -y build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /var/task

# Upgrade pip and install pipenv
RUN pip install --upgrade pip pipenv

# Copy Pipfile and Pipfile.lock first to take advantage of caching
COPY Pipfile Pipfile.lock ./

# Generate and install requirements
RUN pipenv requirements > requirements.txt && \
    pip install -r requirements.txt --no-cache-dir

# Download NLP models
RUN python -m spacy download en_core_web_sm && \
    python -m nltk.downloader -d /usr/share/nltk_data punkt punkt_tab stopwords wordnet

# Final stage
FROM ghcr.io/abetlen/llama-cpp-python:v0.3.1

WORKDIR /var/task

# Copy installed Python packages, executables, and data from builder
COPY --from=builder /usr/local /usr/local
COPY --from=builder /usr/share/nltk_data /usr/share/nltk_data/

# Copy application code
COPY app /var/task/app
COPY icon.png /var/task/

# Set environment variables
ENV PYTHONPATH=/var/task
ENV TOKENIZER_PATH=/var/task/app/models/tokenizer.pkl

# Use Uvicorn with workers for better concurrency
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]
