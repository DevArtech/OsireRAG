"""
Module: embedding.py

Contains everything to embed Document objects and strings of text (queries).

Classes:
- EmbeddedChunk: Pydantic model for a chunk of text with an embedding (inherits: Chunk).
- DocumentEmbedder: Pydantic model for an embedder of documents and text.

Functions:
- None

Usage:
- Import the embedding object from this module to use the DocumentEmbedder class (ensures the model is only loaded once).

Author: Adam Haile  
Date: 10/9/2024
"""

from pydantic import BaseModel
from typing import List, ClassVar
from langchain_huggingface import HuggingFaceEmbeddings

from app.core.settings import get_settings
from app.core.logger import logger, COLORS
from app.core.models.chunker import Chunk


class EmbeddedChunk(Chunk):
    """
    A Pydantic model for a chunk of text with an embedding (inherits: Chunk).

    Attributes:
    - embedding: List[float]: The embedding of the chunk.

    Methods:
    - None

    Usage:
    - Create an instance of this class to represent a chunk of text with an embedding.

    Author: Adam Haile  
    Date: 10/9/2024
    """

    embedding: List[float]


class DocumentEmbedder(BaseModel):
    """
    A Pydantic model for an embedder of documents and text.

    Attributes:
    - hf: HuggingFaceEmbeddings: The HuggingFaceEmbeddings object.

    Methods:
    - embed_chunks: Embeds a list of chunks.
    - embed_query: Embeds a string (query).

    Usage:
    - Use the pre-initialized embedder object to embed chunks and queries.

    Author: Adam Haile  
    Date: 10/9/2024
    """

    hf: ClassVar[HuggingFaceEmbeddings] = None

    class Config:
        arbitrary_types_allowed = True

    def __init__(self):
        """
        A Pydantic model for an embedder of documents and text.

        Attributes:
        - hf: HuggingFaceEmbeddings: The HuggingFaceEmbeddings object.

        Methods:
        - embed_chunks: Embeds a list of chunks.
        - embed_query: Embeds a string (query).

        Usage:
        - Use the pre-initialized embedder object to embed chunks and queries.

        Author: Adam Haile  
        Date: 10/9/2024
        """
        super().__init__()
        logger.warning(
            f"{COLORS().WARNING}Initializing Embedding Model - This may take a second{COLORS().RESET}"
        )
        # Initialize the HuggingFaceEmbeddings object
        DocumentEmbedder.hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": get_settings().DEVICE if get_settings().DEVICE else "cuda"
            },
            encode_kwargs={"normalize_embeddings": False},
        )
        logger.info(f"{COLORS().INFO}Embedding Model Initialized{COLORS().RESET}")

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        """
        Embed a list of Chunk objects.

        Args:
        - `chunks (List[Chunk])`: The list of Chunk objects to embed.

        Returns:
        - List[EmbeddedChunk]: The list of EmbeddedChunk objects with embeddings.

        Usage:
        - `embedded_chunks = embedder.embed_chunks(chunks)`

        Author: Adam Haile  
        Date: 10/9/2024
        """
        embedded_docs = self.hf.embed_documents([chunk.content for chunk in chunks])
        return [
            EmbeddedChunk(embedding=embedded_docs[i], **chunk.model_dump())
            for i, chunk in enumerate(chunks)
        ]

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a string (query).

        Args:
        - `query (str)`: The query to embed.

        Returns:
        - List[float]: The embedding of the query.

        Usage:
        - `embedded_query = embedder.embed_query(query)`

        Author: Adam Haile  
        Date: 10/9/2024
        """
        return self.hf.embed_query(query)


embedder = DocumentEmbedder()