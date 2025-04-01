"""
Unit tests for the KnowledgeBase class.

This module contains test cases for the KnowledgeBase class, testing its initialization,
component classes, and methods for managing and searching a knowledge base.

Author: Adam Haile
Date: 3/31/2024
"""

import os
import pytest
import tempfile
from unittest.mock import patch, MagicMock, mock_open
from fastapi import UploadFile
from io import BytesIO

from app.core.models.knowledge_base import KnowledgeBase, SearchParameters, DocumentArgs
from app.core.models.chunker import Chunk
from app.core.models.documents import Document

# Test data constants
TEST_PROJECT = "test_project"
TEST_VECTORSTORE = "test_vectorstore"
TEST_MODEL = "test_model"
TEST_QUERY = "test query"


@pytest.fixture
def mock_file():
    """Create a mock UploadFile for testing."""
    content = b"This is test content"
    file = MagicMock(spec=UploadFile)
    file.filename = "test.txt"
    file.file = BytesIO(content)
    return file


@pytest.fixture
def test_chunks():
    """Create test chunks for testing."""
    return [
        Chunk(content=f"Test document {i}", index=i, metadata={"id": f"doc{i}"})
        for i in range(5)
    ]


@pytest.fixture
def test_kb():
    """Create a test KnowledgeBase with mocked components."""
    with patch("app.core.models.knowledge_base.DocumentChunker") as mock_chunker, \
         patch("app.core.models.knowledge_base.WebScraper") as mock_scraper, \
         patch("app.core.models.knowledge_base.DocumentEmbedder") as mock_embedder, \
         patch("app.core.models.knowledge_base.VectorstoreManager") as mock_vs_manager, \
         patch("app.core.models.knowledge_base.ChunkTokenizer") as mock_tokenizer, \
         patch("app.core.models.knowledge_base.BM25Model") as mock_bm25, \
         patch("app.core.models.knowledge_base.ReciprocalRankFusion") as mock_rrf, \
         patch("app.core.models.knowledge_base.Reranker") as mock_reranker, \
         patch("app.core.models.knowledge_base.embedder") as mock_embedder_instance:

        # Return a KnowledgeBase with mocked components
        kb = KnowledgeBase()
        
        # Set up mocked components for testing
        kb.chunker = MagicMock()
        kb.scraper = MagicMock()
        kb.embedder = MagicMock()
        kb.vs_manager = MagicMock()
        kb.tokenizer = MagicMock()
        kb.bm25 = MagicMock()
        kb.rrf = MagicMock()
        kb.reranker = MagicMock()
        
        yield kb


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    with tempfile.TemporaryDirectory() as temp_path:
        old_cwd = os.getcwd()
        os.chdir(temp_path)
        
        # Create needed directories
        os.makedirs(os.path.join(".osirerag", TEST_PROJECT, TEST_VECTORSTORE), exist_ok=True)
        os.makedirs(os.path.join(".osirerag", TEST_PROJECT, TEST_MODEL), exist_ok=True)
        
        yield temp_path
        
        os.chdir(old_cwd)


def test_search_parameters_initialization():
    """Test SearchParameters initialization with default values."""
    params = SearchParameters(query=TEST_QUERY)
    
    assert params.query == TEST_QUERY
    assert params.n_results == 10
    assert params.filter == {}
    assert params.rerank is True
    assert params.allow_no_results is True
    assert params.threshold is None


def test_search_parameters_custom_values():
    """Test SearchParameters initialization with custom values."""
    params = SearchParameters(
        query=TEST_QUERY,
        n_results=5,
        filter={"source": "test"},
        rerank=False,
        allow_no_results=False,
        threshold=0.75
    )
    
    assert params.query == TEST_QUERY
    assert params.n_results == 5
    assert params.filter == {"source": "test"}
    assert params.rerank is False
    assert params.allow_no_results is False
    assert params.threshold == 0.75


def test_document_args_initialization():
    """Test DocumentArgs initialization with default values."""
    args = DocumentArgs(
        project_name=TEST_PROJECT,
        vectorstore_name=TEST_VECTORSTORE,
        model_name=TEST_MODEL
    )
    
    assert args.project_name == TEST_PROJECT
    assert args.vectorstore_name == TEST_VECTORSTORE
    assert args.model_name == TEST_MODEL
    assert args.n == 7
    assert args.chunk_len == 10000
    assert args.chunk_overlap == 50
    assert args.k1 == 1.5
    assert args.b == 0.75
    assert args.epsilon == 0.25


def test_document_args_custom_values():
    """Test DocumentArgs initialization with custom values."""
    args = DocumentArgs(
        project_name=TEST_PROJECT,
        vectorstore_name=TEST_VECTORSTORE,
        model_name=TEST_MODEL,
        n=5,
        chunk_len=5000,
        chunk_overlap=100,
        k1=1.2,
        b=0.8,
        epsilon=0.3
    )
    
    assert args.project_name == TEST_PROJECT
    assert args.vectorstore_name == TEST_VECTORSTORE
    assert args.model_name == TEST_MODEL
    assert args.n == 5
    assert args.chunk_len == 5000
    assert args.chunk_overlap == 100
    assert args.k1 == 1.2
    assert args.b == 0.8
    assert args.epsilon == 0.3


def test_knowledge_base_initialization():
    """Test KnowledgeBase initialization."""
    kb = KnowledgeBase()
    
    assert hasattr(kb, "chunker")
    assert hasattr(kb, "scraper")
    assert hasattr(kb, "embedder")
    assert hasattr(kb, "vs_manager")
    assert hasattr(kb, "tokenizer")
    assert hasattr(kb, "bm25")
    assert hasattr(kb, "rrf")
    assert hasattr(kb, "reranker")


def test_validate_project_existing(test_kb, temp_dir):
    """Test _validate_project with existing project."""
    # Project already exists in the temp_dir
    project_path, vs_path, model_path = test_kb._validate_project(
        TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL
    )
    
    assert TEST_PROJECT in project_path
    assert TEST_VECTORSTORE in vs_path
    assert TEST_MODEL in model_path


def test_validate_project_create(test_kb):
    """Test _validate_project creating a new project."""
    with patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as mock_makedirs:
        
        test_kb._validate_project(
            TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, create_if_not_exists=True
        )
        
        # Check that directories were created
        assert mock_makedirs.call_count == 3


def test_validate_project_not_found(test_kb):
    """Test _validate_project with non-existent project."""
    with patch("os.path.exists", return_value=False):
        with pytest.raises(ValueError, match="Project not found"):
            test_kb._validate_project(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL)


def test_validate_project_already_exists(test_kb):
    """Test _validate_project with already existing project."""
    with patch("os.path.exists", return_value=True):
        with pytest.raises(ValueError, match="Project already exists"):
            test_kb._validate_project(
                TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, create_if_not_exists=True
            )


def test_create_kb(test_kb):
    """Test create_kb method."""
    with patch.object(test_kb, "_validate_project") as mock_validate:
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Create mock vectorstore
        mock_vectorstore = MagicMock()
        test_kb.vs_manager.create_vectorstore.return_value = mock_vectorstore
        
        # Call create_kb
        test_kb.create_kb(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL)
        
        # Check that _validate_project was called with create_if_not_exists=True
        mock_validate.assert_called_once_with(
            TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, create_if_not_exists=True
        )
        
        # Check that create_vectorstore and save_vectorstore were called
        test_kb.vs_manager.create_vectorstore.assert_called_once()
        test_kb.vs_manager.save_vectorstore.assert_called_once_with(
            mock_vectorstore, "vs_path"
        )


def test_add_documents(test_kb, mock_file):
    """Test add_documents method."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file_open:
        
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Mock BM25 load_model to raise FileNotFoundError (new model)
        test_kb.bm25.load_model.side_effect = FileNotFoundError()
        
        # Create mock for document chunks and embeddings
        mock_chunks = [MagicMock() for _ in range(3)]
        mock_embeddings = [MagicMock() for _ in range(3)]
        mock_tokens = [MagicMock() for _ in range(3)]
        
        # Set up return values for mocked components
        test_kb.vs_manager.load_vectorstore.return_value = MagicMock()
        test_kb.chunker.chunk_document.return_value = mock_chunks
        test_kb.embedder.embed_chunks.return_value = mock_embeddings
        test_kb.tokenizer.tokenize_documents.return_value = mock_tokens
        test_kb.vs_manager.add_chunks.return_value = ["id1", "id2", "id3"]
        
        # Mock Document creation
        with patch("app.core.models.knowledge_base.Document") as mock_document_cls:
            mock_document_cls.return_value = MagicMock()
            
            # Call add_documents
            args = DocumentArgs(
                project_name=TEST_PROJECT,
                vectorstore_name=TEST_VECTORSTORE,
                model_name=TEST_MODEL
            )
            
            result = test_kb.add_documents(args, [mock_file])
            
            # Check that file was written
            mock_file_open.assert_called()
            
            # Check that document was processed correctly
            test_kb.chunker.chunk_document.assert_called_once()
            test_kb.embedder.embed_chunks.assert_called_once_with(mock_chunks)
            test_kb.tokenizer.tokenize_documents.assert_called_once_with(mock_chunks)
            test_kb.vs_manager.add_chunks.assert_called_once()
            test_kb.vs_manager.save_vectorstore.assert_called_once()
            test_kb.bm25.create_model.assert_called_once()
            
            # Check result
            assert result == ["id1", "id2", "id3"]


def test_add_documents_existing_file(test_kb, mock_file):
    """Test add_documents with existing file."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=True):
        
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Mock BM25 load_model to return a valid tuple
        test_kb.bm25.load_model.return_value = ([], MagicMock())
        
        # Call add_documents
        args = DocumentArgs(
            project_name=TEST_PROJECT,
            vectorstore_name=TEST_VECTORSTORE,
            model_name=TEST_MODEL
        )
        
        # Should raise ValueError due to existing file
        with pytest.raises(ValueError, match="Document of same name already exists"):
            test_kb.add_documents(args, [mock_file])


def test_add_documents_with_existing_model(test_kb, mock_file):
    """Test add_documents method with existing BM25 model."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=False), \
         patch("os.makedirs") as mock_makedirs, \
         patch("builtins.open", mock_open()) as mock_file_open:
        
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Mock existing documents and model
        existing_docs = [MagicMock() for _ in range(2)]
        existing_model = MagicMock()
        test_kb.bm25.load_model.return_value = (existing_docs, existing_model)
        
        # Create mock for document chunks and embeddings
        mock_chunks = [MagicMock() for _ in range(3)]
        mock_embeddings = [MagicMock() for _ in range(3)]
        mock_tokens = [MagicMock() for _ in range(3)]
        
        # Set up return values for mocked components
        test_kb.vs_manager.load_vectorstore.return_value = MagicMock()
        test_kb.chunker.chunk_document.return_value = mock_chunks
        test_kb.embedder.embed_chunks.return_value = mock_embeddings
        test_kb.tokenizer.tokenize_documents.return_value = mock_tokens
        test_kb.vs_manager.add_chunks.return_value = ["id1", "id2", "id3"]
        
        # Mock Document creation
        with patch("app.core.models.knowledge_base.Document") as mock_document_cls:
            mock_document_cls.return_value = MagicMock()
            
            # Call add_documents
            args = DocumentArgs(
                project_name=TEST_PROJECT,
                vectorstore_name=TEST_VECTORSTORE,
                model_name=TEST_MODEL
            )
            
            result = test_kb.add_documents(args, [mock_file])
            
            # Check that existing docs were used
            test_kb.bm25.create_model.assert_called_once()
            
            # The token_docs parameter should include both existing and new tokens
            create_model_args = test_kb.bm25.create_model.call_args[0]
            assert create_model_args[0] == TEST_PROJECT
            assert create_model_args[1] == TEST_MODEL
            # The third argument should be a list that includes both existing_docs and new tokens
            assert len(create_model_args[2]) == len(existing_docs) + len(mock_tokens)


def test_add_webpages(test_kb):
    """Test add_webpages method."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("app.core.models.knowledge_base.KnowledgeBase.add_documents") as mock_add_documents:
        
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Mock the scraper to return mock files
        mock_files = [MagicMock() for _ in range(2)]
        test_kb.scraper.add_pages.return_value = mock_files
        
        # Mock add_documents to return IDs
        mock_add_documents.return_value = ["id1", "id2"]
        
        # Call add_webpages
        args = DocumentArgs(
            project_name=TEST_PROJECT,
            vectorstore_name=TEST_VECTORSTORE,
            model_name=TEST_MODEL
        )
        
        result = test_kb.add_webpages(args, ["https://example.com"])
        
        # Check that validate_project was called
        mock_validate.assert_called_once()
        
        # Check that scraper was called
        test_kb.scraper.add_pages.assert_called_once_with(TEST_PROJECT, ["https://example.com"])
        
        # Check that add_documents was called with scraped files
        mock_add_documents.assert_called_once_with(args, mock_files, upload=False)
        
        # Check result
        assert result == ["id1", "id2"]


def test_search(test_kb, test_chunks):
    """Test search method."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=True), \
         patch("os.listdir") as mock_listdir:
        
        # Mock the return value of _validate_project
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        
        # Mock listdir to return some documents plus vectorstore and model
        mock_listdir.return_value = ["doc1.txt", "doc2.txt", TEST_VECTORSTORE, TEST_MODEL]
        
        # Create mock for vectorstore, BM25 model and search results
        mock_vectorstore = MagicMock()
        mock_bm25_model = MagicMock()
        mock_token_chunks = [MagicMock() for _ in range(3)]
        
        # Mock query embeddings and tokens
        mock_query_embedding = MagicMock()
        mock_query_tokens = ["test", "query"]
        
        # Mock search results
        mock_faiss_results = [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(test_chunks[:3])]
        mock_keyword_results = [(chunk, 0.8 - i * 0.1) for i, chunk in enumerate(test_chunks[2:])]
        mock_rrf_results = [(chunk, 0.95 - i * 0.05) for i, chunk in enumerate(test_chunks[:5])]
        
        # Set up return values for mocked components
        test_kb.vs_manager.load_vectorstore.return_value = mock_vectorstore
        test_kb.bm25.load_model.return_value = (mock_token_chunks, mock_bm25_model)
        test_kb.embedder.embed_query.return_value = mock_query_embedding
        test_kb.tokenizer.tokenize_query.return_value = mock_query_tokens
        test_kb.vs_manager.search.return_value = mock_faiss_results
        test_kb.bm25.search.return_value = mock_keyword_results
        test_kb.rrf.ranks.return_value = mock_rrf_results
        
        # If rerank=True, reranker will be called
        test_kb.reranker.cross_encode_rerank.return_value = [(chunk, 0.98 - i * 0.08) for i, chunk in enumerate(test_chunks[:3])]
        
        # Call search
        params = SearchParameters(query=TEST_QUERY, n_results=5, rerank=True)
        result = test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params)
        
        # Check that validate_project was called
        mock_validate.assert_called_once()
        
        # Check that vectorstore and BM25 model were loaded
        test_kb.vs_manager.load_vectorstore.assert_called_once_with("vs_path")
        test_kb.bm25.load_model.assert_called_once_with(TEST_PROJECT, TEST_MODEL)
        
        # Check that query was processed
        test_kb.embedder.embed_query.assert_called_once_with(TEST_QUERY)
        test_kb.tokenizer.tokenize_query.assert_called_once_with(TEST_QUERY)
        
        # Check that searches were performed
        test_kb.vs_manager.search.assert_called_once()
        test_kb.bm25.search.assert_called_once_with(mock_query_tokens, mock_bm25_model, mock_token_chunks, k=5)
        
        # Check that RRF was called
        test_kb.rrf.ranks.assert_called_once_with([mock_faiss_results, mock_keyword_results], n=5, threshold=None)
        
        # Check that reranker was called
        test_kb.reranker.cross_encode_rerank.assert_called_once()
        
        # Check result
        assert result == test_kb.reranker.cross_encode_rerank.return_value


def test_search_with_no_rerank(test_kb, test_chunks):
    """Test search method with rerank=False."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=True), \
         patch("os.listdir") as mock_listdir:
        
        # Setup similar to test_search, but with rerank=False
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        mock_listdir.return_value = ["doc1.txt", "doc2.txt", TEST_VECTORSTORE, TEST_MODEL]
        
        mock_vectorstore = MagicMock()
        mock_bm25_model = MagicMock()
        mock_token_chunks = [MagicMock() for _ in range(3)]
        
        mock_query_embedding = MagicMock()
        mock_query_tokens = ["test", "query"]
        
        mock_faiss_results = [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(test_chunks[:3])]
        mock_keyword_results = [(chunk, 0.8 - i * 0.1) for i, chunk in enumerate(test_chunks[2:])]
        mock_rrf_results = [(chunk, 0.95 - i * 0.05) for i, chunk in enumerate(test_chunks[:5])]
        
        test_kb.vs_manager.load_vectorstore.return_value = mock_vectorstore
        test_kb.bm25.load_model.return_value = (mock_token_chunks, mock_bm25_model)
        test_kb.embedder.embed_query.return_value = mock_query_embedding
        test_kb.tokenizer.tokenize_query.return_value = mock_query_tokens
        test_kb.vs_manager.search.return_value = mock_faiss_results
        test_kb.bm25.search.return_value = mock_keyword_results
        test_kb.rrf.ranks.return_value = mock_rrf_results
        
        # Call search with rerank=False
        params = SearchParameters(query=TEST_QUERY, n_results=5, rerank=False)
        result = test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params)
        
        # Reranker should not be called
        test_kb.reranker.cross_encode_rerank.assert_not_called()
        
        # Result should be directly from RRF
        assert result == mock_rrf_results


def test_search_no_documents(test_kb):
    """Test search with no documents in project."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=True), \
         patch("os.listdir") as mock_listdir:
        
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        # Only return vectorstore and model directories, no documents
        mock_listdir.return_value = [TEST_VECTORSTORE, TEST_MODEL]
        
        # Call search with allow_no_results=True
        params = SearchParameters(query=TEST_QUERY, allow_no_results=True)
        result = test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params)
        
        # Should return empty list without error
        assert result == []
        
        # Call search with allow_no_results=False
        params = SearchParameters(query=TEST_QUERY, allow_no_results=False)
        with pytest.raises(ValueError, match="No documents found in project"):
            test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params)


def test_search_with_threshold(test_kb, test_chunks):
    """Test search method with threshold parameter."""
    with patch.object(test_kb, "_validate_project") as mock_validate, \
         patch("os.path.exists", return_value=True), \
         patch("os.listdir") as mock_listdir:
        
        # Setup similar to test_search
        mock_validate.return_value = ("project_path", "vs_path", "model_path")
        mock_listdir.return_value = ["doc1.txt", "doc2.txt", TEST_VECTORSTORE, TEST_MODEL]
        
        mock_vectorstore = MagicMock()
        mock_bm25_model = MagicMock()
        mock_token_chunks = [MagicMock() for _ in range(3)]
        
        mock_query_embedding = MagicMock()
        mock_query_tokens = ["test", "query"]
        
        mock_faiss_results = [(chunk, 0.9 - i * 0.1) for i, chunk in enumerate(test_chunks[:3])]
        mock_keyword_results = [(chunk, 0.8 - i * 0.1) for i, chunk in enumerate(test_chunks[2:])]
        
        test_kb.vs_manager.load_vectorstore.return_value = mock_vectorstore
        test_kb.bm25.load_model.return_value = (mock_token_chunks, mock_bm25_model)
        test_kb.embedder.embed_query.return_value = mock_query_embedding
        test_kb.tokenizer.tokenize_query.return_value = mock_query_tokens
        test_kb.vs_manager.search.return_value = mock_faiss_results
        test_kb.bm25.search.return_value = mock_keyword_results
        
        # Set a threshold
        threshold = 0.7
        params = SearchParameters(query=TEST_QUERY, threshold=threshold, rerank=False)
        test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params)
        
        # Check that RRF was called with the threshold
        test_kb.rrf.ranks.assert_called_once_with([mock_faiss_results, mock_keyword_results], n=10, threshold=threshold)


def test_search_nonexistent_project(test_kb):
    """Test search with nonexistent project."""
    with patch.object(test_kb, "_validate_project") as mock_validate:
        # Mock validate_project to raise ValueError
        mock_validate.side_effect = ValueError("Project not found")
        
        params = SearchParameters(query=TEST_QUERY)
        with pytest.raises(ValueError, match="Project not found"):
            test_kb.search(TEST_PROJECT, TEST_VECTORSTORE, TEST_MODEL, params) 