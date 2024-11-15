from typing import List
from langchain_huggingface import HuggingFaceEmbeddings

from core.settings import get_settings
from core.logger import logger, COLORS
from core.models.chunker import Chunk


class EmbeddedChunk(Chunk):
    embedding: List[float]


class DocumentEmbedder:
    if get_settings().MODE != "lightweight":
        logger.warning(
            f"{COLORS().WARNING}Initializing Embedding Model - This may take a second{COLORS().RESET}"
        )
        hf = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2",
            model_kwargs={
                "device": get_settings().DEVICE if get_settings().DEVICE else "cuda"
            },
            encode_kwargs={"normalize_embeddings": False},
        )
        logger.info(f"{COLORS().INFO}Embedding Model Initialized{COLORS().RESET}")
    else:
        hf = None

    def embed_chunks(self, chunks: List[Chunk]) -> List[EmbeddedChunk]:
        if get_settings().MODE == "lightweight" and not self.hf:
            logger.warning(
                f"{COLORS().WARNING}Initializing Embedding Model - This may take a second"
            )
            self.hf = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": get_settings().DEVICE if get_settings().DEVICE else "cuda"
                },
                encode_kwargs={"normalize_embeddings": False},
            )
            logger.info(f"{COLORS().INFO}Embedding Model Initialized")

        embedded_docs = self.hf.embed_documents([chunk.content for chunk in chunks])
        return [
            EmbeddedChunk(embedding=embedded_docs[i], **chunk.model_dump())
            for i, chunk in enumerate(chunks)
        ]

    def embed_query(self, query: str) -> List[float]:
        if get_settings().MODE == "lightweight" and not self.hf:
            logger.warning(
                f"{COLORS().WARNING}Initializing Embedding Model - This may take a second"
            )
            self.hf = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                model_kwargs={
                    "device": get_settings().DEVICE if get_settings().DEVICE else "cuda"
                },
                encode_kwargs={"normalize_embeddings": False},
            )
            logger.info(f"{COLORS().INFO}Embedding Model Initialized")

        return self.hf.embed_query(query)
