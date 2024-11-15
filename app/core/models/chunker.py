import os
import warnings
from typing import Any
from spacy.errors import Warnings


warnings.filterwarnings("ignore", message=r"\[W095\]")

import spacy
from typing import List, Dict
from pydantic import BaseModel
from langchain_text_splitters import (
    HTMLHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)

from core.settings import get_settings
from core.models.documents import Document

headers_to_split_on = [
    ("h1", "Header 1"),
    ("h2", "Header 2"),
    ("h3", "Header 3"),
]


class Chunk(BaseModel):
    content: str
    index: int
    metadata: Dict[str, Any] = {}

    def __hash__(self):
        return hash((self.index, self.content))


class DocumentChunker:
    def __init__(self) -> None:
        if get_settings().MODE != "lightweight":
            self.nlp = spacy.load("en_core_web_sm")
            self.splitter = HTMLHeaderTextSplitter(headers_to_split_on)
        else:
            self.nlp = None
            self.splitter = None

    def chunk_document(self, document: Document, n: int = 3, **kwargs) -> List[Chunk]:
        if get_settings().MODE == "lightweight" and (not self.nlp or not self.splitter):
            self.nlp = spacy.load("en_core_web_sm")
            self.splitter = HTMLHeaderTextSplitter(headers_to_split_on)

        max_char_length = kwargs.get("max_length", 10000)
        overlap = kwargs.get("overlap", 50)

        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_char_length,
            chunk_overlap=overlap,
            length_function=len,
            is_separator_regex=False,
        )

        metadata = {"directory": os.path.abspath(document.directory).replace("\\", "/")}
        if document.directory.endswith(".pdf") or document.directory.endswith(".txt"):
            merged_chunks = []
            chunks = recursive_splitter.split_text(document.content)
            for chunk in chunks:
                sentences = [
                    str(sentence).strip() for sentence in self.nlp(chunk).sents
                ]
                for i in range(0, len(sentences), n):
                    merged_chunks.append(" ".join(sentences[i : i + n]))

            return [
                Chunk(content=chunk, index=i, metadata=metadata)
                for i, chunk in enumerate(merged_chunks)
            ]

        if document.directory.endswith(".html"):
            splits = self.splitter.split_text(document.content)

            merged_chunks = []
            splits = [split.page_content for split in splits]
            for i in range(0, len(sentences), n):
                chunks = " ".join(sentences[i : i + n])
                merged_chunks.append(chunks)

            return [
                Chunk(content=chunk, index=i, metadata=metadata)
                for i, chunk in enumerate(merged_chunks)
            ]

        return []
