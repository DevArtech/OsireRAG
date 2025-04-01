"""
Unit tests for the Document class.

This module contains test cases for the Document class, testing its initialization,
validation, and property access methods.

Author: Adam Haile
Date: 3/31/2024
"""

import os
import pytest
from app.core.models.documents import Document

# Test data directory - you'll need to create this and add test files
TEST_DATA_DIR = os.path.join(os.path.dirname(__file__), "test_data")

def setup_module(module):
    """Create test data directory and files if they don't exist."""
    if not os.path.exists(TEST_DATA_DIR):
        os.makedirs(TEST_DATA_DIR)
    
    # Create test files
    with open(os.path.join(TEST_DATA_DIR, "test.txt"), "w", encoding="utf-8") as f:
        f.write("This is a test text file.")
    
    with open(os.path.join(TEST_DATA_DIR, "test.html"), "w", encoding="utf-8") as f:
        f.write("<html><body>This is a test HTML file.</body></html>")

def teardown_module(module):
    """Clean up test files."""
    for file in ["test.txt", "test.html"]:
        file_path = os.path.join(TEST_DATA_DIR, file)
        if os.path.exists(file_path):
            os.remove(file_path)
    if os.path.exists(TEST_DATA_DIR):
        os.rmdir(TEST_DATA_DIR)

def test_document_initialization_with_txt():
    """Test Document initialization with a text file."""
    file_path = os.path.join(TEST_DATA_DIR, "test.txt")
    doc = Document(directory=file_path)
    
    # Normalize both paths for comparison
    expected_path = file_path.replace("\\", "/")
    actual_path = doc.directory.replace("\\", "/")
    assert actual_path == expected_path
    assert doc.content == "This is a test text file."
    assert doc.metadata == {}

def test_document_initialization_with_html():
    """Test Document initialization with an HTML file."""
    file_path = os.path.join(TEST_DATA_DIR, "test.html")
    doc = Document(directory=file_path)
    
    # Normalize both paths for comparison
    expected_path = file_path.replace("\\", "/")
    actual_path = doc.directory.replace("\\", "/")
    assert actual_path == expected_path
    assert doc.content == "<html><body>This is a test HTML file.</body></html>"
    assert doc.metadata == {}

def test_document_path_normalization():
    """Test that file paths are normalized correctly."""
    file_path = os.path.join(TEST_DATA_DIR, "test.txt")
    normalized_path = file_path.replace("\\", "/")
    doc = Document(directory=file_path)
    
    assert doc.directory.replace("\\", "/") == normalized_path

def test_invalid_file_type():
    """Test that invalid file types raise ValueError."""
    invalid_file = os.path.join(TEST_DATA_DIR, "test.invalid")
    with open(invalid_file, "w") as f:
        f.write("test")
    
    with pytest.raises(ValueError, match="File .invalid type not supported"):
        Document(directory=invalid_file)
    
    os.remove(invalid_file)

def test_nonexistent_file():
    """Test that nonexistent files raise ValueError."""
    with pytest.raises(ValueError, match='"./nonexistent.txt" directory does not exist'):
        Document(directory="./nonexistent.txt")

def test_path_traversal():
    """Test that path traversal attempts are blocked."""
    with pytest.raises(ValueError) as exc_info:
        Document(directory="../test.txt")
    assert "Path traversal" in str(exc_info.value)

def test_content_property():
    """Test the content property access."""
    file_path = os.path.join(TEST_DATA_DIR, "test.txt")
    doc = Document(directory=file_path)
    
    assert doc.content == "This is a test text file."

def test_metadata_property():
    """Test the metadata property access."""
    file_path = os.path.join(TEST_DATA_DIR, "test.txt")
    doc = Document(directory=file_path)
    
    assert isinstance(doc.metadata, dict)
    assert doc.metadata == {} 