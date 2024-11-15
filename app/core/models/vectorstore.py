import os
import uuid
import faiss
import numpy as np
import pickle
from pydantic import BaseModel
from typing import List, Dict, Optional, Tuple, Any, Union

from core.models.chunker import Chunk
from core.models.embedding import EmbeddedChunk, DocumentEmbedder


class VectorstoreSearchParameters(BaseModel):
    embedded_query: List[float]
    k: int = 10
    filter: Optional[Dict[str, str]] = None


class FAISS(BaseModel):
    embedding_function: Any  # HuggingFaceEmbeddings
    index: Any  # faiss.IndexFlatL2
    docstore: Dict[Any, Any]
    index_to_docstore_id: Dict[int, Any]


class VectorstoreManager:
    def __init__(self):
        self.embedding_function = DocumentEmbedder().hf
        self.index = faiss.IndexFlatL2(384)
        self.docstore = {}

    def create_vectorstore(self) -> FAISS:
        return FAISS(
            embedding_function=self.embedding_function,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={},
        )

    def add_chunks(
        self, vectorstore: Union[str, FAISS], chunks: List[EmbeddedChunk]
    ) -> List[str]:
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        vectorstore.index.add(embeddings)
        for i, chunk in enumerate(chunks):
            doc_id = ids[i]
            vectorstore.docstore[doc_id] = {
                "content": chunk.content,
                "index": chunk.index,
                **chunk.metadata,
                "embedding": chunk.embedding,
            }
            vectorstore.index_to_docstore_id[i] = doc_id

        return ids

    def get_chunks(self, vectorstore: Union[str, FAISS], ids: List[str]) -> List[Chunk]:
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        chunks = []
        for doc_id in ids:
            doc_data = vectorstore.docstore.get(doc_id)
            if doc_data:
                chunks.append(
                    Chunk(
                        content=doc_data["content"],
                        index=doc_data["index"],
                        metadata={
                            k: v
                            for k, v in doc_data.items()
                            if k not in {"content", "index", "embedding"}
                        },
                    )
                )
        return chunks

    def delete_chunks(self, vectorstore: Union[str, FAISS], ids: List[str]) -> None:
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        indices_to_delete = [
            idx
            for idx, doc_id in vectorstore.index_to_docstore_id.items()
            if doc_id in ids
        ]
        for doc_id in ids:
            if doc_id in vectorstore.docstore:
                del vectorstore.docstore[doc_id]

        for idx in indices_to_delete:
            del vectorstore.index_to_docstore_id[idx]

        if indices_to_delete:
            new_index = faiss.IndexFlatL2(vectorstore.index.d)
            remaining_embeddings = [
                vectorstore.docstore[vectorstore.index_to_docstore_id[idx]]["embedding"]
                for idx in vectorstore.index_to_docstore_id
            ]
            if remaining_embeddings:
                new_index.add(np.array(remaining_embeddings, dtype=np.float32))
            vectorstore.index = new_index

    def search(
        self, vectorstore: Union[str, FAISS], search_params: VectorstoreSearchParameters
    ) -> List[Tuple[Chunk, float]]:
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        query = np.array([search_params.embedded_query], dtype=np.float32)
        distances, indices = vectorstore.index.search(query, search_params.k)

        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = vectorstore.index_to_docstore_id[idx]
            doc_data = vectorstore.docstore[doc_id]
            metadata = {
                k: v
                for k, v in doc_data.items()
                if k not in {"content", "embedding", "index"}
            }
            chunk = Chunk(
                content=doc_data["content"],
                index=doc_data["index"],
                metadata=metadata,
            )
            results.append((chunk, dist))

        return results

    def save_vectorstore(self, vectorstore: FAISS, path: str) -> None:
        # Ensure the directory exists
        os.makedirs(path, exist_ok=True)

        # Save the FAISS index to path/index.faiss
        faiss.write_index(vectorstore.index, f"{path}/index.faiss")

        # Save the docstore and index_to_docstore_id as path/index.pkl
        with open(f"{path}/index.pkl", "wb") as f:
            pickle.dump(
                {
                    "docstore": vectorstore.docstore,
                    "index_to_docstore_id": vectorstore.index_to_docstore_id,
                },
                f,
            )

    def load_vectorstore(self, path: str) -> FAISS:
        # Load the FAISS index from path/index.faiss
        index = faiss.read_index(f"{path}/index.faiss")

        # Load the docstore and index_to_docstore_id from path/index.pkl
        with open(f"{path}/index.pkl", "rb") as f:
            data = pickle.load(f)
            docstore = data["docstore"]
            index_to_docstore_id = data["index_to_docstore_id"]

        # Return a new FAISS instance with loaded components
        return FAISS(
            embedding_function=self.embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
