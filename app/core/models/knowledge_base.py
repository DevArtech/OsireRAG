import os
from pydantic import BaseModel
from fastapi import UploadFile
from typing import Dict, Any, List, Tuple

from core.models.documents import Document
from core.models.chunker import DocumentChunker, Chunk
from core.models.embedding import DocumentEmbedder
from core.models.vectorstore import VectorstoreManager, VectorstoreSearchParameters
from core.models.term_freq_retriever import ChunkTokenizer, BM25Model
from core.models.rrf import ReciprocalRankFusion
from core.models.reranker import Reranker


class SearchParameters(BaseModel):
    query: str
    n_results: int = 10
    filter: Dict[str, Any] = {}
    rerank: bool = True


class KnowledgeBase:
    def __init__(self):
        self.chunker = DocumentChunker()
        self.embedder = DocumentEmbedder()
        self.vs_manager = VectorstoreManager()
        self.tokenizer = ChunkTokenizer()
        self.bm25 = BM25Model()
        self.rrf = ReciprocalRankFusion()
        self.reranker = Reranker()

    def create_kb(self, project_name: str, vectorstore_name: str, model_name: str):
        project_path = f"./.rosierag/{project_name}"
        if not os.path.exists(project_path):
            os.makedirs(project_path)

        vs_path = f"./.rosierag/{project_name}/{vectorstore_name}"
        if os.path.exists(vs_path):
            raise ValueError("Vectorstore already exists.")

        model_path = f"./.rosierag/{project_name}/{model_name}"
        if os.path.exists(model_path):
            raise ValueError("Keyword model already exists.")

        os.mkdir(model_path)

        vectorstore = self.vs_manager.create_vectorstore()
        self.vs_manager.save_vectorstore(vectorstore, vs_path)

    def add_documents(
        self,
        project_name: str,
        vectorstore_name: str,
        model_name: str,
        documents: List[UploadFile],
    ):
        project_path = f"./.rosierag/{project_name}"
        if not os.path.exists(project_path):
            raise ValueError("Project not found.")

        vs_path = f"./.rosierag/{project_name}/{vectorstore_name}"
        if not os.path.exists(vs_path):
            raise ValueError("Vectorstore does not exists.")

        model_path = f"./.rosierag/{project_name}/{model_name}"
        if not os.path.exists(model_path):
            raise ValueError("Keyword model does not exists.")

        ids = []
        tokenized_docs = []
        for document in documents:
            file_path = project_path + f"/{document.filename}"
            if os.path.exists(file_path):
                raise ValueError(
                    f"Document of same name already exists: {document.filename}"
                )

            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "wb") as f:
                f.write(document.file.read())

            vectorstore = self.vs_manager.load_vectorstore(vs_path)

            document = Document(directory=os.path.abspath(file_path))
            document_chunks = self.chunker.chunk_document(document)
            document_embeddings = self.embedder.embed_chunks(document_chunks)
            document_tokens = self.tokenizer.tokenize_documents(document_chunks)
            tokenized_docs.extend(document_tokens)

            ids.extend(self.vs_manager.add_chunks(vectorstore, document_embeddings))

        self.vs_manager.save_vectorstore(vectorstore, vs_path)

        self.bm25.create_model(project_name, model_name, tokenized_docs)

        return ids

    def add_project(self, project_name: str, vectorstore_name: str):
        project_path = f"./.rosierag/{project_name}"
        if not os.path.exists(project_path):
            raise ValueError("Project not found.")

        kb_path = f"{project_path}/{vectorstore_name}"
        if not os.path.exists(kb_path):
            raise ValueError("Knowledge base not found.")

        documents = os.listdir(project_path)
        if not documents:
            raise ValueError("No documents found in project.")

        vectorstore = self.vs_manager.load_vectorstore(kb_path)

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
        self.vs_manager.save_vectorstore(vectorstore, kb_path)

    def search(
        self,
        project_name: str,
        vectorstore_name: str,
        model_name: str,
        params: SearchParameters,
    ) -> List[Tuple[Chunk, float]]:
        project_path = f"./.rosierag/{project_name}"
        if not os.path.exists(project_path):
            raise ValueError("Project not found.")

        vs_path = f"./.rosierag/{project_name}/{vectorstore_name}"
        if not os.path.exists(vs_path):
            raise ValueError("Vectorstore does not exists.")

        model_path = f"./.rosierag/{project_name}/{model_name}"
        if not os.path.exists(model_path):
            raise ValueError("Keyword model does not exists.")

        vectorstore = self.vs_manager.load_vectorstore(vs_path)
        token_chunks, bm25_model = self.bm25.load_model(project_name, model_name)

        query_embeddings = self.embedder.embed_query(params.query)
        query_tokens = self.tokenizer.tokenize_query(params.query)

        vs_params = VectorstoreSearchParameters(
            embedded_query=query_embeddings, k=params.n_results, filter=params.filter
        )

        faiss_chunks = self.vs_manager.search(vectorstore, vs_params)
        keyword_chunks = self.bm25.search(
            query_tokens, bm25_model, token_chunks, k=params.n_results
        )

        chunks = self.rrf.ranks([faiss_chunks, keyword_chunks], n=params.n_results)

        if params.rerank:
            chunks = self.reranker.cross_encode_rerank(
                params.query, [chunk for chunk, score in chunks]
            )

        return chunks
