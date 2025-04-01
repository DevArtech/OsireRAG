"""
Unit tests for the WebScraper class.

This module contains test cases for the WebScraper class, testing its web page
downloading and file saving functionality.

Author: Adam Haile
Date: 3/31/2024
"""

import os
import pytest
import tempfile
import shutil
from unittest.mock import patch, MagicMock, mock_open
from fastapi import UploadFile
from fastapi.responses import JSONResponse
from app.core.models.web import WebScraper

# Create a temporary directory structure that matches the expected path in WebScraper
@pytest.fixture
def setup_project_dir():
    """Setup the project directory structure needed for tests."""
    # Create a temporary directory for the tests
    temp_dir = tempfile.mkdtemp()
    
    # Create the .osirerag directory structure
    os.makedirs(os.path.join(temp_dir, ".osirerag", "test_project"), exist_ok=True)
    
    # Patch the os.path.exists to return True for our test project
    original_exists = os.path.exists
    
    def mock_exists(path):
        if path == "./.osirerag/test_project":
            return True
        elif path == "./.osirerag/nonexistent_project":
            return False
        else:
            return original_exists(path)
    
    with patch('os.path.exists', side_effect=mock_exists):
        yield
        
    # Clean up
    shutil.rmtree(temp_dir)

@pytest.fixture
def mock_file_operations():
    """Mock file operations to avoid actual file creation."""
    with patch('builtins.open', mock_open()):
        with patch('os.makedirs') as mock_makedirs:
            with patch('os.listdir', return_value=[]):
                yield mock_makedirs

def test_add_pages_success(setup_project_dir, mock_file_operations):
    """Test successful web page download and file creation."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <head>
            <title>Test Page</title>
        </head>
        <body>
            <h1>Test Content</h1>
        </body>
    </html>
    """
    mock_response.content = b"<html><body>Test Content</body></html>"

    with patch('requests.get', return_value=mock_response):
        with patch('builtins.open', mock_open()):
            with patch('io.BytesIO') as mock_bytes_io:
                scraper = WebScraper()
                result = scraper.add_pages("test_project", ["https://example.com"])

                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], UploadFile)
                assert result[0].filename == "test-page.html"

def test_add_pages_no_title(setup_project_dir, mock_file_operations):
    """Test web page download when no title is found."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "<html><body>No title</body></html>"
    mock_response.content = b"<html><body>No title</body></html>"

    with patch('requests.get', return_value=mock_response):
        with patch('builtins.open', mock_open()):
            with patch('io.BytesIO'):
                with patch('os.listdir', return_value=[]):
                    scraper = WebScraper()
                    result = scraper.add_pages("test_project", ["https://example.com"])

                    assert isinstance(result, list)
                    assert len(result) == 1
                    assert isinstance(result[0], UploadFile)
                    assert result[0].filename.endswith(".html")
                    assert "example-1" in result[0].filename or "1" in result[0].filename

def test_add_pages_invalid_project(setup_project_dir):
    """Test adding pages to non-existent project."""
    scraper = WebScraper()
    result = scraper.add_pages("nonexistent_project", ["https://example.com"])

    assert isinstance(result, JSONResponse)
    assert result.status_code == 404
    assert "Project not found" in result.body.decode()

def test_add_pages_download_failure(setup_project_dir):
    """Test handling of failed web page download."""
    mock_response = MagicMock()
    mock_response.status_code = 404
    mock_response.text = "Not Found"

    with patch('requests.get', return_value=mock_response):
        scraper = WebScraper()
        result = scraper.add_pages("test_project", ["https://example.com"])

        assert isinstance(result, JSONResponse)
        assert result.status_code == 404
        assert "Failed to download page" in result.body.decode()

def test_add_pages_sanitize_filename(setup_project_dir, mock_file_operations):
    """Test filename sanitization for special characters."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = """
    <html>
        <head>
            <title>Test Page with Special Chars: *?<>:"/\\|</title>
        </head>
        <body>
            <h1>Test Content</h1>
        </body>
    </html>
    """
    mock_response.content = b"<html><body>Test Content</body></html>"

    with patch('requests.get', return_value=mock_response):
        with patch('builtins.open', mock_open()):
            with patch('io.BytesIO'):
                scraper = WebScraper()
                result = scraper.add_pages("test_project", ["https://example.com"])

                assert isinstance(result, list)
                assert len(result) == 1
                assert isinstance(result[0], UploadFile)
                assert result[0].filename == "test-page-with-special-chars-.html" 