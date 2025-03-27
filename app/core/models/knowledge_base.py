"""
Module: knowledge_base.py

Contains the KnowledgeBase of the OsireRAG system. Utilizes all submodules to create a centralized knowledge base.

Classes:
- KnowledgeBase: A class representing the OsireRAG knowledge base.
- SearchParameters: Dataclass model for search parameters.
- DocumentArgs: Dataclass model for document arguments.

Functions:
- None

Usage:
- Import the KnowledgeBase class from this module into other modules that require a knowledge base.

Author: Adam Haile  
Date: 10/16/2024
"""

import os
import time
from pydantic import BaseModel
from dataclasses import dataclass, field
from fastapi import UploadFile
from typing import Dict, Any, List, Tuple

from app.core.logger import logger
from app.core.models.documents import Document
from app.core.models.web import WebScraper
from app.core.models.chunker import DocumentChunker, Chunk
from app.core.models.embedding import DocumentEmbedder, embedder
from app.core.models.vectorstore import VectorstoreManager, VectorstoreSearchParameters
from app.core.models.term_freq_retriever import ChunkTokenizer, BM25Model
from app.core.models.rrf import ReciprocalRankFusion
from app.core.models.reranker import Reranker


@dataclass
class SearchParameters:
    """
    Dataclass model for search parameters.

    Attributes:
    - query: str: The search query.
    - n_results: int: The number of results to return.
    - filter: Dict[str, Any]: A filter for the search.
    - rerank: bool: Whether to rerank the results.

    Methods:
    - None

    Usage:
    - Create an instance of this class to represent search parameters.

    Author: Adam Haile
    Date: 10/16/2024
    """

    query: str
    n_results: int = 10
    filter: Dict[str, Any] = field(default_factory=dict)
    rerank: bool = True


@dataclass
class DocumentArgs:
    """
    Dataclass model for document arguments.

    Attributes:
    - project_name: str: The project name.
    - vectorstore_name: str: The vectorstore name.
    - model_name: str: The model name.
    - n: int: The number of sentences per chunk.

    Methods:
    - None

    Usage:
    - Create an instance of this class to represent document arguments.

    Author: Adam Haile
    Date: 10/16/2024
    """

    project_name: str
    vectorstore_name: str
    model_name: str
    n: int = 7
    chunk_len: int = 10000
    chunk_overlap: int = 50
    k1: float = 1.5
    b: float = 0.75
    epsilon: float = 0.25


class KnowledgeBase(BaseModel):
    """
    A class representing the OsireRAG knowledge base.

    Attributes:
    - chunker: DocumentChunker: The document chunker.
    - scraper: WebScraper: The web scraper.
    - embedder: DocumentEmbedder: The document embedder.
    - vs_manager: VectorstoreManager: The vectorstore manager.
    - tokenizer: ChunkTokenizer: The chunk tokenizer.
    - bm25: BM25Model: The BM25 model.
    - rrf: ReciprocalRankFusion: The reciprocal rank fusion model.
    - reranker: Reranker: The reranker model.

    Methods:
    - _validate_project: Validates a project exists.
    - create_kb: Creates a knowledge base.
    - add_documents: Adds documents to the knowledge base.
    - add_webpages: Adds webpages to the knowledge base.
    - add_project: Adds an entire project to the knowledge base.
    - search: Searches the knowledge base.

    Usage:
    - Create an instance of this class to represent the OsireRAG knowledge base.

    Author: Adam Haile
    Date: 10/16/2024
    """

    chunker: DocumentChunker = DocumentChunker()
    scraper: WebScraper = WebScraper()
    embedder: DocumentEmbedder = embedder
    vs_manager: VectorstoreManager = VectorstoreManager()
    tokenizer: ChunkTokenizer = ChunkTokenizer()
    bm25: BM25Model = BM25Model()
    rrf: ReciprocalRankFusion = ReciprocalRankFusion()
    reranker: Reranker = Reranker()

    def _validate_project(
        self, project: str, vs: str, model: str, create_if_not_exists: bool = False
    ) -> Tuple[str, str, str]:
        """
        Validates a project exists.

        Args:
        - `project (str)`: The project name.
        - `vs (str)`: The vectorstore name.
        - `model (str)`: The model name.
        - `create_if_not_exists (bool)`: Whether to create the project if it does not exist.

        Returns:
        - Tuple[str, str, str]: The project path, vectorstore path, and model path.

        Raises:
        - ValueError: If the project does not exist (or already exists if create_if_not_exists = True).

        Usage:
        - Use this method to validate a project exists before performing operations on it.
        - `project_path, vs_path, model_path = self._validate_project("project", "vectorstore", "model")`

        Author: Adam Haile
        Date: 10/16/2024
        """

        # Validate the project
        project_path = os.path.join(os.getcwd(), ".osirerag", project)
        if not os.path.exists(project_path):
            if create_if_not_exists:
                os.makedirs(project_path)
            else:
                raise ValueError("Project not found.")
        elif create_if_not_exists:
            raise ValueError("Project already exists.")

        # Validate the vectorstore
        vs_path = os.path.join(os.getcwd(), ".osirerag", project, vs)
        if not os.path.exists(vs_path):
            if create_if_not_exists:
                os.makedirs(vs_path)
            else:
                raise ValueError("Vectorstore does not exists.")
        elif create_if_not_exists:
            raise ValueError("Vectorstore already exists")

        # Validate the keyword model
        model_path = os.path.join(os.getcwd(), ".osirerag", project, model)
        if not os.path.exists(model_path):
            if create_if_not_exists:
                os.makedirs(model_path)
            else:
                raise ValueError("Keyword model does not exists.")
        elif create_if_not_exists:
            raise ValueError("Keyword model already exists")

        # Return the paths of each validated item
        return project_path, vs_path, model_path

    def create_kb(
        self, project_name: str, vectorstore_name: str, model_name: str
    ) -> None:
        """
        Creates a knowledge base.

        Args:
        - `project_name (str)`: The project name.
        - `vectorstore_name (str)`: The vectorstore name.
        - `model_name (str)`: The model name.

        Returns:
        - None

        Raises:
        - ValueError: If the project already exists.

        Usage:
        - `kbase.create_kb("project", "vectorstore", "model")`

        Author: Adam Haile
        Date: 10/16/2024
        """
        _, vs_path, _ = self._validate_project(
            project_name, vectorstore_name, model_name, create_if_not_exists=True
        )

        # Create and save a new vectorstore to the vs_path
        # (Same process is not necessary for the keyword model, model is saved when data is added)
        vectorstore = self.vs_manager.create_vectorstore()
        self.vs_manager.save_vectorstore(vectorstore, vs_path)

    def add_documents(
        self,
        args: DocumentArgs,
        documents: List[UploadFile],
        upload: bool = True,
    ) -> List[str]:
        """
        Adds documents to the knowledge base.

        Args:
        - `args (DocumentArgs)`: The document arguments.
        - `documents (List[UploadFile])`: The list of documents to add.
        - `upload (bool)`: Whether to upload the documents or use pre-added ones.

        Returns:
        - List[str]: The list of document IDs.

        Raises:
        - ValueError: If a document of the same name already exists.

        Usage:
        - `kbase.add_documents(args, documents)`

        Author: Adam Haile
        Date: 10/16/2024
        """
        # Track the time it takes to add the documents
        start_time = time.time()

        # Validate the project
        project_path, vs_path, _ = self._validate_project(
            args.project_name, args.vectorstore_name, args.model_name
        )

        ids = []
        tokenized_docs = []
        new_tokenized_docs = []

        # Attempt to load the existing BM25 model and documents
        try:
            existing_docs, _ = self.bm25.load_model(args.project_name, args.model_name)
            tokenized_docs.extend(existing_docs)
            logger.info("Existing BM25 model and documents loaded.")
        except FileNotFoundError:
            # If no existing model, initialize a new list
            logger.info("No existing BM25 model found. Creating a new one.")

        for document in documents:
            # Save the document to the project directory
            file_path = os.path.join(project_path, document.filename)
            if upload:
                # Check if a document of the same name already exists
                if os.path.exists(file_path):
                    raise ValueError(
                        f"Document of same name already exists: {document.filename}"
                    )

                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, "wb") as f:
                    f.write(document.file.read())

            # Load the vectorstore
            logger.info(f"Processing document: {document.filename}")
            vectorstore = self.vs_manager.load_vectorstore(vs_path)

            # Convert the document to a Document object
            document_obj = Document(directory=os.path.abspath(file_path))

            # Chunk, embed, and tokenize the document
            logger.info(
                f'Chunking document into chunks of sentence size: "{args.n}", len: "{args.chunk_len}", overlap: "{args.chunk_overlap}".'
            )
            document_chunks = self.chunker.chunk_document(
                document_obj,
                n=args.n,
                max_length=args.chunk_len,
                overlap=args.chunk_overlap,
            )

            logger.info("Embedding document.")
            document_embeddings = self.embedder.embed_chunks(document_chunks)

            logger.info("Tokenizing document.")
            document_tokens = self.tokenizer.tokenize_documents(document_chunks)
            new_tokenized_docs.extend(document_tokens)

            # Add the document embeddings to the vectorstore
            ids.extend(self.vs_manager.add_chunks(vectorstore, document_embeddings))
            logger.info(f"{document.filename} preprocessed.")

        # Combine existing and new tokenized documents
        tokenized_docs.extend(new_tokenized_docs)

        # Save the vectorstore and create the new/updated BM25 model
        logger.info("Saving databases and models.")
        self.vs_manager.save_vectorstore(vectorstore, vs_path)
        self.bm25.create_model(args.project_name, args.model_name, tokenized_docs, k1=args.k1, b=args.b, epsilon=args.epsilon)

        # Log the time it took to add the documents
        logger.info(f"Documents added in {time.time() - start_time} seconds.")

        return ids

    def add_webpages(
        self,
        args: DocumentArgs,
        webpages: List[str],
    ) -> List[str]:
        """
        Add webpages to the knowledge base.

        Args:
        - `project_name (str)`: The project name.
        - `vectorstore_name (str)`: The vectorstore name.
        - `model_name (str)`: The model name.

        Returns:
        - List[str]: The list of document IDs.

        Raises:
        - ValueError: If no documents are found in the project.

        Usage:
        - `kbase.add_webpages("project", "vectorstore", "model", ["https://www.somewebsite.com", "https://www.somewebsite2.com"])`

        Author: Adam Haile
        Date: 10/16/2024
        """
        # Validate the project
        self._validate_project(
            args.project_name, args.vectorstore_name, args.model_name
        )

        # Scrape the webpages and run add_documents on the returned HTML files
        logger.info("Downloading webpages to project directory.")
        page_files = self.scraper.add_pages(args.project_name, webpages)
        return self.add_documents(
            args,
            page_files,
            upload=False,
        )

    def add_project(
        self, project_name: str, vectorstore_name: str, model_name: str
    ) -> None:
        project_path, vs_path, _ = self._validate_project(
            project_name, vectorstore_name, model_name
        )

        documents = os.listdir(project_path)
        if not documents:
            raise ValueError("No documents found in project.")

        vectorstore = self.vs_manager.load_vectorstore(vs_path)

        document_chunks = []
        for document in documents:
            document_path = os.path.abspath(project_path + "/" + document).replace(
                "\\", "/"
            )
            if not os.path.isdir(document_path):
                document = Document(directory=document_path)
                document_chunks.extend(self.chunker.chunk_document(document))

        document_embeddings = self.embedder.embed_chunks(document_chunks)
        self.vs_manager.add_chunks(vectorstore, document_embeddings)
        self.vs_manager.save_vectorstore(vectorstore, vs_path)

    def search(
        self,
        project_name: str,
        vectorstore_name: str,
        model_name: str,
        params: SearchParameters,
    ) -> List[Tuple[Chunk, float]]:
        """
        Searches the knowledge base.

        Args:
        - `project_name (str)`: The project name.
        - `vectorstore_name (str)`: The vectorstore name.
        - `model_name (str)`: The model name.

        Returns:
        - List[Tuple[Chunk, float]]: The search results and the score of the Chunk.

        Raises:
        - ValueError: If the project does not exist.

        Usage:
        - `kbase.search("project", "vectorstore", "model", SearchParameters(query="query", n_results=10, filter={}, rerank=True))`

        Author: Adam Haile
        Date: 10/16/2024
        """

        # Track the time it takes to search the knowledge base
        start_time = time.time()

        # Validate the project
        _, vs_path, _ = self._validate_project(
            project_name, vectorstore_name, model_name
        )

        # Load the vectorstore, token chunks, and BM25 model
        vectorstore = self.vs_manager.load_vectorstore(vs_path)
        token_chunks, bm25_model = self.bm25.load_model(project_name, model_name)

        # Embed and tokenize the query
        query_embeddings = self.embedder.embed_query(params.query)
        query_tokens = self.tokenizer.tokenize_query(params.query)

        # Establish the search parameters for the vectorstore
        vs_params = VectorstoreSearchParameters(
            embedded_query=query_embeddings, k=params.n_results, filter=params.filter
        )

        # Search the vectorstore and BM25 model
        faiss_chunks = self.vs_manager.search(vectorstore, vs_params)
        keyword_chunks = self.bm25.search(
            query_tokens, bm25_model, token_chunks, k=params.n_results
        )

        # Rank the results using reciprocal rank fusion
        chunks = self.rrf.ranks([faiss_chunks, keyword_chunks], n=params.n_results)

        # Rerank the results if necessary
        if params.rerank:
            chunks = self.reranker.cross_encode_rerank(
                params.query, [chunk for chunk, score in chunks]
            )

        # Log the time it took to search the knowledge base
        logger.info(f"Search completed in {time.time() - start_time} seconds.")

        return chunks
