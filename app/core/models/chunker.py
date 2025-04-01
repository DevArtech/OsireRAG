"""
Module: chunker.py

Classes:
- Chunk: Pydantic model for a chunk of text.
- DocumentChunker: Pydantic model for a document chunker.

Functions:
- None

Usage:
- Import the DocumentChunker class from this module into the main FastAPI app.

Author: Adam Haile  
Date: 10/16/2024
"""

import os
import warnings
from typing import Any
from spacy.errors import Warnings
from spacy.language import Language


warnings.filterwarnings("ignore", message=r"\[W095\]")

import spacy
from pydantic import BaseModel, ConfigDict
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from typing import List, Dict, ClassVar

from app.core.settings import get_settings
from app.core.models.documents import Document

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]


class Chunk(BaseModel):
    """
    Pydantic model for a chunk of text.

    Attributes:
    - content: str: The content of the chunk.
    - index: int: The index of the chunk.
    - metadata: Dict[str, Any]: Metadata for the chunk.

    Methods:
    - __hash__: Hashes the chunk.

    Usage:
    - Create an instance of this class to represent a chunk of text.

    Author: Adam Haile
    Date: 10/9/2024
    """

    content: str
    index: int
    metadata: Dict[str, Any] = {}

    def __hash__(self):
        return hash((self.index, self.content))


class DocumentChunker(BaseModel):
    """
    Pydantic model for a document chunker.

    Attributes:
    - nlp: Language: The spaCy language model.
    - splitter: HTMLHeaderTextSplitter: The splitter for HTML headers.

    Methods:
    - chunk_document: Chunks a document into smaller pieces.

    Author: Adam Haile
    Date: 10/9/2024
    """

    # Use ConfigDict to replace the deprecated Config class
    model_config = ConfigDict(arbitrary_types_allowed=True)
    
    nlp: ClassVar[Language] = spacy.load("en_core_web_sm")
    splitter: ClassVar[HTMLHeaderTextSplitter] = HTMLHeaderTextSplitter(
        headers_to_split_on
    )

    def chunk_document(self, document: Document, n: int = 3, **kwargs) -> List[Chunk]:
        """
        Chunks a document into smaller pieces.

        Args:
        - `document (Document)`: The document to chunk.
        - `n (int)`: The number of sentences to include in each chunk.

        Returns:
        - List[Chunk]: The chunks of the document.

        Usage:
        - `chunks = chunk_document(document, n=7)`

        Author: Adam Haile
        Date: 10/9/2024
        """

        # Additional optional kwargs for splitting text
        max_char_length = kwargs.get("max_length", 10000)
        overlap = kwargs.get("overlap", 50)

        # Instantiate the RecursiveCharacterTextSplitter
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_char_length,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )

        # Set the metadata for the chunks
        metadata = {"directory": os.path.abspath(document.directory).replace("\\", "/")}

        # Validate the document type and split accordingly
        if document.directory.endswith(".pdf") or document.directory.endswith(".txt"):
            merged_chunks = []

            # Split the document into chunks of n sentences
            chunks = recursive_splitter.split_text(document.content)

            for chunk in chunks:
                # Split the chunk into sentences
                sentences = [
                    str(sentence).strip() for sentence in self.nlp(chunk).sents
                ]

                # Merge the sentences into chunks of n sentences
                for i in range(0, len(sentences), n):
                    merged_chunks.append(" ".join(sentences[i : i + n]))

            return [
                Chunk(content=chunk, index=i, metadata=metadata)
                for i, chunk in enumerate(merged_chunks)
            ]

        if document.directory.endswith(".html"):
            # Split the document into chunks of n sentences
            splits = self.splitter.split_text(document.content)

            # Get the content of each split
            splits = [split.page_content for split in splits]

            return [
                Chunk(content=chunk, index=i, metadata=metadata)
                for i, chunk in enumerate(splits)
            ]

        return []
