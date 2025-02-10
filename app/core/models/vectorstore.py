"""
Module: vectorstore.py

Classes:
- VectorstoreSearchParameters: A Pydantic model for search parameters used in vectorstore queries.
- FAISS: A Pydantic model representing a FAISS-based vectorstore.
- VectorstoreManager: A Pydantic model for managing vectorstores with chunk embeddings.

Functions:
- None

Usage:
- Import the VectorstoreManager class to manage vectorstores, add or retrieve chunks, and perform searches.

Author: Adam Haile  
Date: 10/13/2024
"""

import os
import uuid
import faiss
import pickle
import numpy as np
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from typing import List, Dict, Optional, Tuple, Any, Union, ClassVar

from app.core.models.chunker import Chunk
from app.core.models.embedding import EmbeddedChunk, embedder


class VectorstoreSearchParameters(BaseModel):
    """
    A Pydantic model for search parameters used in vectorstore queries.

    Attributes:
    - embedded_query: List[float]: The embedding of the query.
    - k: int: The number of top results to return. Default is 10.
    - filter: Optional[Dict[str, str]]: A dictionary for filtering results. Default is None.

    Usage:
    - Instantiate this class to specify search parameters for vectorstore queries.

    Author: Adam Haile
    Date: 10/13/2024
    """

    embedded_query: List[float]
    k: int = 10
    filter: Optional[Dict[str, str]] = None


class FAISS(BaseModel):
    """
    A Pydantic model representing a FAISS-based vectorstore.

    Attributes:
    - embedding_function: HuggingFaceEmbeddings: The embedding function used to create embeddings.
    - index: faiss.IndexFlatL2: The FAISS index for efficient similarity searches.
    - docstore: Dict[Any, Any]: A dictionary mapping document IDs to metadata.
    - index_to_docstore_id: Dict[int, Any]: A mapping from index positions to document IDs.

    Usage:
    - Use this class to represent a vectorstore and its associated data.

    Author: Adam Haile
    Date: 10/13/2024
    """

    embedding_function: HuggingFaceEmbeddings
    index: faiss.IndexFlatL2
    docstore: Dict[Any, Any]
    index_to_docstore_id: Dict[int, Any]

    class Config:
        arbitrary_types_allowed = True


class VectorstoreManager(BaseModel):
    """
    A Pydantic model for managing vectorstores with chunk embeddings.

    Attributes:
    - embedding_function: Any: The embedding function to use for creating embeddings.
    - index: Any: The FAISS index for managing embeddings.
    - docstore: Dict[Any, Any]: A dictionary for storing document metadata.

    Methods:
    - create_vectorstore: Creates a new FAISS vectorstore.
    - add_chunks: Adds chunks with embeddings to the vectorstore.
    - get_chunks: Retrieves chunks from the vectorstore by their IDs.
    - delete_chunks: Deletes chunks from the vectorstore by their IDs.
    - search: Searches the vectorstore using an embedded query.
    - save_vectorstore: Saves the vectorstore to disk.
    - load_vectorstore: Loads a vectorstore from disk.

    Usage:
    - Instantiate this class to manage vectorstores and perform operations on them.

    Author: Adam Haile
    Date: 10/13/2024
    """

    embedding_function: ClassVar[HuggingFaceEmbeddings] = embedder.hf
    index: ClassVar[faiss.IndexFlatL2] = faiss.IndexFlatL2(384)
    docstore: Dict[Any, Any] = {}

    class Config:
        arbitrary_types_allowed = True

    def create_vectorstore(self) -> FAISS:
        """
        Creates a new FAISS vectorstore.

        Returns:
        - FAISS: A new FAISS vectorstore instance.

        Usage:
        - vectorstore = manager.create_vectorstore()

        Author: Adam Haile
        Date: 10/13/2024
        """
        return FAISS(
            embedding_function=self.embedding_function,
            index=self.index,
            docstore=self.docstore,
            index_to_docstore_id={},
        )

    def add_chunks(
        self, vectorstore: Union[str, FAISS], chunks: List[EmbeddedChunk]
    ) -> List[str]:
        """
        Adds chunks with embeddings to the vectorstore.

        Args:
        - `vectorstore (Union[str, FAISS])`: The vectorstore or its path.
        - `chunks (List[EmbeddedChunk])`: The chunks to add.

        Returns:
        - List[str]: A list of IDs for the added chunks.

        Usage:
        - `chunk_ids = manager.add_chunks(vectorstore, embedded_chunks)`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Load the vectorstore if it's a path
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        # Initialize the IDs and get the embeddings
        ids = [str(uuid.uuid4()) for _ in range(len(chunks))]
        embeddings = np.array([chunk.embedding for chunk in chunks], dtype=np.float32)

        # Get the start index for the new embeddings
        start_idx = vectorstore.index.ntotal

        # Add the embeddings to the index
        vectorstore.index.add(embeddings)

        # Add the chunks to the docstore
        for i, chunk in enumerate(chunks):
            doc_id = ids[i]
            vectorstore.docstore[doc_id] = {
                "content": chunk.content,
                "index": chunk.index,
                **chunk.metadata,
                "embedding": chunk.embedding,
            }
            vectorstore.index_to_docstore_id[start_idx + i] = doc_id

        return ids

    def get_chunks(self, vectorstore: Union[str, FAISS], ids: List[str]) -> List[Chunk]:
        """
        Retrieves chunks from the vectorstore by their IDs.

        Args:
        - `vectorstore (Union[str, FAISS])`: The vectorstore or its path.
        - `ids (List[str])`: A list of chunk IDs.

        Returns:
        - List[Chunk]: The retrieved chunks.

        Usage:
        - `chunks = manager.get_chunks(vectorstore, chunk_ids)`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Load the vectorstore if it's a path
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        chunks = []
        # Get the chunks from the docstore
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
        """
        Deletes chunks from the vectorstore by their IDs.

        Args:
        - `vectorstore (Union[str, FAISS])`: The vectorstore or its path.
        - `ids (List[str])`: A list of chunk IDs.

        Returns:
        - None

        Usage:
        - `manager.delete_chunks(vectorstore, chunk_ids)`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Load the vectorstore if it's a path
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        # Get the indices to delete
        indices_to_delete = [
            idx
            for idx, doc_id in vectorstore.index_to_docstore_id.items()
            if doc_id in ids
        ]

        # Delete the chunks from the docstore and index
        for doc_id in ids:
            if doc_id in vectorstore.docstore:
                del vectorstore.docstore[doc_id]

        for idx in indices_to_delete:
            del vectorstore.index_to_docstore_id[idx]

        # Rebuild the index if necessary
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
        """
        Searches the vectorstore using an embedded query.

        Args:
        - `vectorstore (Union[str, FAISS])`: The vectorstore or its path.
        - `search_params (VectorstoreSearchParameters)`: The search parameters.

        Returns:
        - List[Tuple[Chunk, float]]: A list of chunks and their distances.

        Usage:
        - `results = manager.search(vectorstore, search_params)`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Load the vectorstore if it's a path
        if isinstance(vectorstore, str):
            vectorstore = self.load_vectorstore(vectorstore)

        # Perform the search
        query = np.array([search_params.embedded_query], dtype=np.float32)
        distances, indices = vectorstore.index.search(query, search_params.k)

        results = []
        # Get the chunks and distances
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            doc_id = vectorstore.index_to_docstore_id[idx]
            doc_data = vectorstore.docstore[doc_id]
            metadata = {  # Extract metadata, excluding specific fields.
                k: v
                for k, v in doc_data.items()
                if k not in {"content", "embedding", "index"}
            }
            chunk = Chunk(
                content=doc_data["content"],
                index=doc_data["index"],
                metadata=metadata,
            )
            results.append((chunk, dist))  # Add the Chunk and its distance to results.

        return results

    def save_vectorstore(self, vectorstore: FAISS, path: str) -> None:
        """
        Saves the vectorstore to disk.

        Args:
        - `vectorstore (FAISS)`: The vectorstore to save.
        - `path (str)`: The directory path to save the vectorstore.

        Returns:
        - None

        Usage:
        - `manager.save_vectorstore(vectorstore, "./path")`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Create the directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save the index and docstore to disk
        faiss.write_index(vectorstore.index, f"{path}/index.faiss")
        with open(f"{path}/index.pkl", "wb") as f:
            pickle.dump(
                {
                    "docstore": vectorstore.docstore,
                    "index_to_docstore_id": vectorstore.index_to_docstore_id,
                },
                f,
            )

    def load_vectorstore(self, path: str) -> FAISS:
        """
        Loads a vectorstore from disk.

        Args:
        - `path (str)`: The directory path to load the vectorstore from.

        Returns:
        - FAISS: The loaded vectorstore.

        Usage:
        - `vectorstore = manager.load_vectorstore("./path")`

        Author: Adam Haile
        Date: 10/13/2024
        """
        # Load the index and docstore from disk
        index = faiss.read_index(f"{path}/index.faiss")
        with open(f"{path}/index.pkl", "rb") as f:
            data = pickle.load(f)
            docstore = data["docstore"]
            index_to_docstore_id = data["index_to_docstore_id"]

        # Return the vectorstore
        return FAISS(
            embedding_function=self.embedding_function,
            index=index,
            docstore=docstore,
            index_to_docstore_id=index_to_docstore_id,
        )
