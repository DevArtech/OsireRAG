import os
import nltk
import pickle
import numpy as np
from typing import List, Tuple
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from core.models.chunker import Chunk


class TokenizedChunk(Chunk):
    tokens: List[str]


class ChunkTokenizer:
    def __init__(self):
        nltk.data.path.append(os.path.abspath("./nltk_data"))
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def tokenize_query(self, query: str) -> List[str]:
        tokens = word_tokenize(query.lower())
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and word not in self.stop_words
        ]
        return tokens

    def tokenize_documents(self, chunks: List[Chunk]) -> List[TokenizedChunk]:
        tokenized_docs = []
        for chunk in chunks:
            tokens = word_tokenize(chunk.content.lower())
            tokens = [
                self.lemmatizer.lemmatize(word)
                for word in tokens
                if word.isalpha() and word not in self.stop_words
            ]
            tokenized_docs.append(TokenizedChunk(tokens=tokens, **chunk.model_dump()))

        return tokenized_docs


class BM25Model:
    def create_model(
        self, project_name: str, model_name: str, chunks: List[TokenizedChunk], **kwargs
    ):
        path = os.path.join("./.rosierag", project_name, model_name)

        if not os.path.exists(os.path.join(path)):
            os.makedirs(path)

        tokenized_corpus = [chunk.tokens for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus, **kwargs)

        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(kwargs, f)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(bm25, f)

    def load_model(
        self, project_name: str, model_name: str
    ) -> Tuple[List[TokenizedChunk], BM25Okapi]:
        path = os.path.join("./.rosierag", project_name, model_name)

        with open(os.path.abspath(os.path.join(path, "documents.pkl")), "rb") as f:
            tokenized_documents = pickle.load(f)

        with open(os.path.abspath(os.path.join(path, "model.pkl")), "rb") as f:
            index = pickle.load(f)

        return tokenized_documents, index

    def search(
        self,
        tokenized_query: List[str],
        model: BM25Okapi,
        chunks: List[TokenizedChunk],
        k: int = 10,
    ) -> List[Tuple[Chunk, float]]:
        scores = model.get_scores(tokenized_query)

        chunks = [Chunk(**chunk.model_dump()) for chunk in chunks]

        scored_docs = [(chunk, score) for chunk, score in zip(chunks, scores)]
        scored_docs.sort(key=lambda x: x[1])
        return scored_docs[::-1][:k]
