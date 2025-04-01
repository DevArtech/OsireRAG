"""
Unit tests for the term_freq_retriever module.

This module contains test cases for the TokenizedChunk, ChunkTokenizer, and BM25Model
classes, testing their initialization and methods.

Author: Adam Haile
Date: 3/31/2024
"""

import os
import pytest
import pickle
import numpy as np
from unittest.mock import patch, MagicMock, mock_open
import tempfile
import shutil

from app.core.models.term_freq_retriever import TokenizedChunk, ChunkTokenizer, BM25Model
from app.core.models.chunker import Chunk

# Test data
TEST_CONTENT = "This is a test document for tokenization and searching."
TEST_METADATA = {"source": "test"}
TEST_TOKENS = ["test", "document", "tokenization", "searching"]
TEST_QUERY = "test searching document"
TEST_TOKENIZED_QUERY = ["test", "search", "document"]
TEST_PROJECT = "test_project"
TEST_MODEL = "test_model"

@pytest.fixture
def mock_nltk():
    """Mock NLTK dependencies."""
    with patch('app.core.models.term_freq_retriever.nltk') as mock_nltk:
        with patch('app.core.models.term_freq_retriever.word_tokenize') as mock_tokenize:
            # Create a lemmatizer instance that we can track
            lemmatizer_instance = MagicMock()
            lemmatizer_instance.lemmatize.side_effect = lambda word: word[:-3] + "ch" if word.endswith('ing') else word
            
            # Mock the WordNetLemmatizer class and its return value
            mock_lemmatizer_cls = MagicMock()
            mock_lemmatizer_cls.return_value = lemmatizer_instance
            
            with patch('app.core.models.term_freq_retriever.WordNetLemmatizer', mock_lemmatizer_cls):
                # Directly patch the ChunkTokenizer class variables
                with patch('app.core.models.term_freq_retriever.ChunkTokenizer.lemmatizer', lemmatizer_instance):
                    with patch('app.core.models.term_freq_retriever.ChunkTokenizer.stop_words', 
                              {"a", "is", "for", "and"}):
                        # Configure word_tokenize mock
                        mock_tokenize.side_effect = lambda text: text.lower().split()
                        
                        yield mock_nltk, mock_tokenize, lemmatizer_instance

@pytest.fixture
def mock_bm25():
    """Mock BM25Okapi."""
    with patch('app.core.models.term_freq_retriever.BM25Okapi') as mock_bm25:
        # Configure mock
        bm25_instance = MagicMock()
        bm25_instance.get_scores.return_value = [0.8, 0.6, 0.4]
        mock_bm25.return_value = bm25_instance
        
        yield mock_bm25, bm25_instance

@pytest.fixture
def test_chunks():
    """Create test chunks."""
    return [
        Chunk(content=f"Test document {i}", index=i, metadata={"id": f"doc{i}"})
        for i in range(3)
    ]

@pytest.fixture
def test_tokenized_chunks():
    """Create test tokenized chunks."""
    return [
        TokenizedChunk(
            content=f"Test document {i}", 
            index=i, 
            metadata={"id": f"doc{i}"},
            tokens=[f"test", "document", f"{i}"]
        )
        for i in range(3)
    ]

@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing file operations."""
    temp_path = tempfile.mkdtemp()
    # Create the expected directory structure
    os.makedirs(os.path.join(temp_path, ".osirerag", TEST_PROJECT, TEST_MODEL), exist_ok=True)
    yield temp_path
    shutil.rmtree(temp_path)

def test_tokenized_chunk_initialization():
    """Test TokenizedChunk initialization."""
    chunk = TokenizedChunk(
        content=TEST_CONTENT,
        index=1,
        metadata=TEST_METADATA,
        tokens=TEST_TOKENS
    )
    
    assert chunk.content == TEST_CONTENT
    assert chunk.index == 1
    assert chunk.metadata == TEST_METADATA
    assert chunk.tokens == TEST_TOKENS
    assert isinstance(chunk, Chunk)  # Test inheritance

def test_chunk_tokenizer_initialization(mock_nltk):
    """Test ChunkTokenizer initialization."""
    mock_nltk_lib, _, _ = mock_nltk
    
    tokenizer = ChunkTokenizer()
    
    # Check that NLTK data path was set
    mock_nltk_lib.data.path.append.assert_called_once()

def test_tokenize_query(mock_nltk):
    """Test tokenizing a query."""
    _, mock_tokenize, mock_lemmatizer = mock_nltk
    
    # Make word_tokenize return tokens that will pass the isalpha() filter
    mock_tokenize.side_effect = lambda text: ["test", "searching", "document"]
    
    tokenizer = ChunkTokenizer()
    tokens = tokenizer.tokenize_query(TEST_QUERY)
    
    # Check that the tokenizer was called
    mock_tokenize.assert_called_once_with(TEST_QUERY.lower())
    
    # Check that lemmatization was applied to each word that passed filters
    assert mock_lemmatizer.lemmatize.call_count == 3  # All 3 tokens should be lemmatized
    
    # Verify the tokens were correctly processed
    assert "test" in tokens  # unchanged
    assert "search" not in tokens  # our mock now transforms "searching" to "search" + "ch"
    assert "search" not in tokens
    assert "searchch" in tokens  # "searching" -> "searchch" with our new mock
    assert "document" in tokens  # unchanged

def test_tokenize_documents(mock_nltk, test_chunks):
    """Test tokenizing documents."""
    _, mock_tokenize, mock_lemmatizer = mock_nltk
    
    # Reset the call count on the lemmatizer to ensure clean testing
    mock_lemmatizer.lemmatize.reset_mock()
    
    # Make word_tokenize return tokens that will pass the isalpha() filter
    # We need a custom side_effect function that returns different tokens for each call
    call_count = 0
    def custom_tokenize(text):
        nonlocal call_count
        tokens = [["test", "document", "first"], 
                 ["test", "document", "second"], 
                 ["test", "document", "searching"]][call_count]
        call_count += 1
        return tokens
    
    mock_tokenize.side_effect = custom_tokenize
    
    tokenizer = ChunkTokenizer()
    tokenized_chunks = tokenizer.tokenize_documents(test_chunks)
    
    # Check that we got the right number of chunks
    assert len(tokenized_chunks) == len(test_chunks)
    
    # Check that all chunks are TokenizedChunk instances
    assert all(isinstance(chunk, TokenizedChunk) for chunk in tokenized_chunks)
    
    # Check that the original content and metadata were preserved
    for i, chunk in enumerate(tokenized_chunks):
        assert chunk.content == test_chunks[i].content
        assert chunk.index == test_chunks[i].index
        assert chunk.metadata == test_chunks[i].metadata
        assert hasattr(chunk, "tokens")
    
    # Check that tokenize was called for each document
    assert mock_tokenize.call_count == len(test_chunks)
    
    # Check that lemmatization was applied to each token in each document
    # We're expecting (3 tokens per document * 3 documents = 9 calls)
    assert mock_lemmatizer.lemmatize.call_count == 9
    
    # Verify the last document has the lemmatized "searchch" token (from "searching")
    assert "searchch" in tokenized_chunks[2].tokens

def test_create_model(mock_bm25, test_tokenized_chunks, temp_dir):
    """Test creating a BM25 model."""
    mock_bm25_cls, mock_bm25_instance = mock_bm25
    
    # Store the original os.path.join function
    original_join = os.path.join
    
    # Patch os.path.join to use our temp directory
    def mock_join(*args):
        # Replace ./.osirerag with our temp dir
        if args[0] == "./.osirerag":
            return original_join(temp_dir, ".osirerag", *args[1:])
        return original_join(*args)
    
    with patch('os.path.join', side_effect=mock_join):
        with patch('builtins.open', mock_open()) as mock_file:
            with patch('pickle.dump') as mock_pickle_dump:
                # Create the model
                bm25_model = BM25Model()
                bm25_model.create_model(TEST_PROJECT, TEST_MODEL, test_tokenized_chunks)
                
                # Check that BM25Okapi was called with the right parameters
                mock_bm25_cls.assert_called_once()
                tokens_list = [chunk.tokens for chunk in test_tokenized_chunks]
                mock_bm25_cls.assert_called_with(tokens_list)
                
                # Check that files were opened for writing
                assert mock_file.call_count == 3
                
                # Check that pickle.dump was called to save the model
                assert mock_pickle_dump.call_count == 3

def test_load_model(test_tokenized_chunks, temp_dir):
    """Test loading a BM25 model."""
    # Create mock data for pickle to load
    mock_bm25_instance = MagicMock()
    
    # Store the original os.path.join function
    original_join = os.path.join
    
    # Patch os.path.join to use our temp directory
    def mock_join(*args):
        # Replace ./.osirerag with our temp dir
        if args[0] == "./.osirerag":
            return original_join(temp_dir, ".osirerag", *args[1:])
        return original_join(*args)
    
    # Create mock pickle data
    documents_data = test_tokenized_chunks
    model_data = mock_bm25_instance
    
    # Track which file is being loaded
    documents_loaded = False
    model_loaded = False
    
    def mock_pickle_load(file_obj):
        nonlocal documents_loaded, model_loaded
        # For the first call, return documents_data
        if not documents_loaded:
            documents_loaded = True
            return documents_data
        # For the second call, return model_data
        elif not model_loaded:
            model_loaded = True
            return model_data
        # Fallback
        return {}
    
    with patch('os.path.join', side_effect=mock_join):
        with patch('os.path.abspath', side_effect=lambda x: x):
            with patch('builtins.open', mock_open()) as mock_file:
                with patch('pickle.load', side_effect=mock_pickle_load):
                    # Load the model
                    bm25_model = BM25Model()
                    loaded_documents, loaded_model = bm25_model.load_model(TEST_PROJECT, TEST_MODEL)
                    
                    # Check that files were opened for reading
                    assert mock_file.call_count == 2
                    
                    # Verify the loaded data
                    assert loaded_documents == documents_data
                    assert loaded_model == model_data

def test_load_model_file_not_found(temp_dir):
    """Test loading a BM25 model when the file does not exist."""
    # Store the original os.path.join function
    original_join = os.path.join
    
    # Patch os.path.join to use our temp directory
    def mock_join(*args):
        # Replace ./.osirerag with our temp dir
        if args[0] == "./.osirerag":
            return original_join(temp_dir, ".osirerag", *args[1:])
        return original_join(*args)
    
    with patch('os.path.join', side_effect=mock_join):
        with patch('os.path.abspath', side_effect=lambda x: x):
            with patch('builtins.open', side_effect=FileNotFoundError):
                # Try to load the model
                bm25_model = BM25Model()
                with pytest.raises(FileNotFoundError):
                    bm25_model.load_model(TEST_PROJECT, TEST_MODEL)

def test_search(test_tokenized_chunks):
    """Test searching with a BM25 model."""
    # Create mock BM25 model and scores
    mock_bm25_model = MagicMock()
    mock_bm25_model.get_scores.return_value = [0.8, 0.6, 0.4]
    
    # Perform search
    bm25 = BM25Model()
    results = bm25.search(
        tokenized_query=TEST_TOKENIZED_QUERY,
        model=mock_bm25_model,
        chunks=test_tokenized_chunks,
        k=2
    )
    
    # Check that get_scores was called with the right query
    mock_bm25_model.get_scores.assert_called_once_with(TEST_TOKENIZED_QUERY)
    
    # Check the results
    assert len(results) == 2  # k=2, so 2 results
    assert all(isinstance(result[0], Chunk) for result in results)
    assert all(isinstance(result[1], float) for result in results)
    
    # Check scores are normalized
    assert 0 <= results[0][1] <= 1
    assert 0 <= results[1][1] <= 1
    
    # First result should have the highest score
    assert results[0][1] > results[1][1]

def test_search_with_zero_scores():
    """Test searching with all zero scores."""
    # Create test chunks
    test_chunks = [
        TokenizedChunk(content=f"Test document {i}", index=i, metadata={}, tokens=["test"])
        for i in range(3)
    ]
    
    # Create mock BM25 model with all zero scores
    mock_bm25_model = MagicMock()
    mock_bm25_model.get_scores.return_value = [0, 0, 0]
    
    # Perform search
    bm25 = BM25Model()
    results = bm25.search(
        tokenized_query=TEST_TOKENIZED_QUERY,
        model=mock_bm25_model,
        chunks=test_chunks,
        k=3
    )
    
    # With all zeros, we should get all chunks with score 1.0
    assert len(results) == 3
    assert all(result[1] == 1.0 for result in results)

def test_search_no_results():
    """Test searching with no results."""
    # Create mock BM25 model with no scores
    mock_bm25_model = MagicMock()
    mock_bm25_model.get_scores.return_value = []
    
    # Perform search with empty chunks list
    bm25 = BM25Model()
    results = bm25.search(
        tokenized_query=TEST_TOKENIZED_QUERY,
        model=mock_bm25_model,
        chunks=[],
        k=5
    )
    
    # Should get empty results list
    assert len(results) == 0 