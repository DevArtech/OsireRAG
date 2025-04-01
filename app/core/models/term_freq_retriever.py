"""
Module: term_freq_retriever.py

Classes:
- TokenizedChunk: A Pydantic model for a text chunk with tokenized content.
- ChunkTokenizer: A Pydantic model for tokenizing queries and document chunks.
- BM25Model: A Pydantic model for creating, loading, and searching BM25 indices.

Functions:
- None

Usage:
- Import the BM25Model and ChunkTokenizer classes from this module to tokenize documents and perform BM25 searches.

Author: Adam Haile  
Date: 10/23/2024
"""

import os
import nltk
import pickle
import numpy as np
from pydantic import BaseModel, ConfigDict
from rank_bm25 import BM25Okapi
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from typing import List, Tuple, Any, ClassVar

from app.core.models.chunker import Chunk


class TokenizedChunk(Chunk):
    """
    A Pydantic model for a chunk of text with tokenized content.

    Attributes:
    - tokens: List[str]: A list of tokens representing the tokenized content of the chunk.

    Usage:
    - Use this class to represent a text chunk after tokenization.

    Author: Adam Haile
    Date: 10/23/2024
    """

    tokens: List[str]


class ChunkTokenizer(BaseModel):
    """
    A Pydantic model for tokenizing queries and document chunks.

    Attributes:
    - stop_words: set: A set of stop words to exclude during tokenization.
    - lemmatizer: Any: A WordNetLemmatizer instance for lemmatizing tokens.

    Methods:
    - tokenize_query: Tokenizes and lemmatizes a query string.
    - tokenize_documents: Tokenizes and lemmatizes the content of document chunks.

    Usage:
    - Instantiate this class to tokenize queries and document chunks for BM25.

    Author: Adam Haile
    Date: 10/23/2024
    """

    stop_words: ClassVar[set] = set(stopwords.words("english"))
    lemmatizer: ClassVar[WordNetLemmatizer] = WordNetLemmatizer()

    # Replace deprecated Config class with ConfigDict
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def __init__(self):
        """
        Initializes the tokenizer with custom NLTK data path.

        Raises:
        - None

        Usage:
        - tokenizer = ChunkTokenizer()

        Author: Adam Haile
        Date: 10/23/2024
        """
        super().__init__()
        nltk.data.path.append(os.path.abspath("./nltk_data"))

    def tokenize_query(self, query: str) -> List[str]:
        """
        Tokenizes and lemmatizes a query string.

        Args:
        - `query (str)`: The query to tokenize.

        Returns:
        - List[str]: A list of lemmatized tokens from the query.

        Usage:
        - `tokens = tokenizer.tokenize_query("example query")`

        Author: Adam Haile
        Date: 10/23/2024
        """
        # Tokenize and lemmatize the query
        tokens = word_tokenize(query.lower())
        tokens = [
            self.lemmatizer.lemmatize(word)
            for word in tokens
            if word.isalpha() and word not in self.stop_words
        ]
        return tokens

    def tokenize_documents(self, chunks: List[Chunk]) -> List[TokenizedChunk]:
        """
        Tokenizes and lemmatizes the content of document chunks.

        Args:
        - `chunks (List[Chunk])`: The list of document chunks to tokenize.

        Returns:
        - List[TokenizedChunk]: A list of TokenizedChunk objects.

        Usage:
        - `tokenized_chunks = tokenizer.tokenize_documents(chunks)`

        Author: Adam Haile
        Date: 10/23/2024
        """
        # Tokenize and lemmatize the content of each chunk
        tokenized_docs = []
        for chunk in chunks:
            tokens = word_tokenize(chunk.content.lower())
            tokens = [
                self.lemmatizer.lemmatize(word)
                for word in tokens
                if word.isalpha() and word not in self.stop_words
            ]
            # Append the tokenized chunk to the list
            tokenized_docs.append(TokenizedChunk(tokens=tokens, **chunk.model_dump()))
        return tokenized_docs


class BM25Model(BaseModel):
    """
    A Pydantic model for creating, loading, and searching BM25 indices.

    Methods:
    - create_model: Creates and saves a BM25 model for a set of tokenized document chunks.
    - load_model: Loads a BM25 model and tokenized documents from disk.
    - search: Performs a BM25 search on a tokenized query.

    Usage:
    - Instantiate this class to create, load, and search BM25 indices.

    Author: Adam Haile
    Date: 10/23/2024
    """

    def create_model(
        self, project_name: str, model_name: str, chunks: List[TokenizedChunk], **kwargs
    ):
        """
        Creates and saves a BM25 model for a set of tokenized document chunks.

        Args:
        - `project_name (str)`: The name of the project.
        - `model_name (str)`: The name of the model.
        - `chunks (List[TokenizedChunk])`: The list of tokenized document chunks.
        - `kwargs (Dict[Any, Any])`: Additional arguments for BM25Okapi.

        Returns:
        - None

        Raises:
        - FileExistsError: If the model directory already exists.

        Usage:
        - `bm25.create_model("project", "model", tokenized_chunks)`
        """
        # Create the model directory if it does not exist
        path = os.path.join("./.osirerag", project_name, model_name)
        os.makedirs(path, exist_ok=True)

        # Create and save the BM25 model from the tokenized documents
        tokenized_corpus = [chunk.tokens for chunk in chunks]
        bm25 = BM25Okapi(tokenized_corpus, **kwargs)

        # Save the tokenized documents and model to disk
        with open(os.path.join(path, "documents.pkl"), "wb") as f:
            pickle.dump(chunks, f)

        with open(os.path.join(path, "data.pkl"), "wb") as f:
            pickle.dump(kwargs, f)

        with open(os.path.join(path, "model.pkl"), "wb") as f:
            pickle.dump(bm25, f)

    def load_model(
        self, project_name: str, model_name: str
    ) -> Tuple[List[TokenizedChunk], BM25Okapi]:
        """
        Loads a BM25 model and tokenized documents from disk.

        Args:
        - `project_name (str)`: The name of the project.
        - `model_name (str)`: The name of the model.

        Returns:
        - Tuple[List[TokenizedChunk], BM25Okapi]: The tokenized documents and the BM25 model.

        Usage:
        - `tokenized_docs, bm25 = bm25.load_model("project", "model")`

        Author: Adam Haile
        Date: 10/23/2024
        """
        # Load the tokenized documents and BM25 model from disk
        path = os.path.join("./.osirerag", project_name, model_name)

        # Load the documents
        try:
            with open(os.path.abspath(os.path.join(path, "documents.pkl")), "rb") as f:
                tokenized_documents = pickle.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"The provided model {model_name} does not have the necessary files. Please ensure documents have been uploaded before attemtping to query the model."
            )

        # Load the BM25 model
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
        """
        Performs a BM25 search on a tokenized query.

        Args:
        - `tokenized_query (List[str])`: The tokenized query.
        - `model (BM25Okapi)`: The BM25 model to use for searching.
        - `chunks (List[TokenizedChunk])`: The list of tokenized document chunks.
        - `k (int)`: The number of top results to return.

        Returns:
        - List[Tuple[Chunk, float]]: A list of document chunks and their normalized BM25 scores.

        Usage:
        - `results = bm25.search(tokenized_query, bm25_model, tokenized_chunks, k=10)`

        Author: Adam Haile
        Date: 10/23/2024
        """
        # Perform the BM25 search
        scores = model.get_scores(tokenized_query)
        
        # Convert chunks to regular Chunk objects
        chunks = [Chunk(**chunk.model_dump()) for chunk in chunks]
        
        # Create list of (chunk, score) pairs
        scored_docs = list(zip(chunks, scores))
        
        # Sort by score in descending order (higher scores first)
        scored_docs.sort(key=lambda x: x[1], reverse=True)
        
        # Take top k results
        top_k = scored_docs[:k]
        
        # Normalize scores to be between 0 and 1
        if top_k:
            max_score = top_k[0][1]
            min_score = top_k[-1][1]
            score_range = max_score - min_score
            
            if score_range > 0:
                normalized_docs = [(doc, (score - min_score) / score_range) for doc, score in top_k]
            else:
                normalized_docs = [(doc, 1.0) for doc, _ in top_k]
        else:
            normalized_docs = []
            
        return normalized_docs
