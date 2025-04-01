"""
Pytest configuration file for OsireRAG tests.

This file contains global pytest configurations and fixtures.

Author: Adam Haile
Date: 3/31/2024
"""

import pytest

# Filter out known warnings
def pytest_configure(config):
    """Configure pytest to ignore specific warnings."""
    # Filter out the FAISS-related numpy warning
    config.addinivalue_line(
        "filterwarnings",
        "ignore:numpy.core._multiarray_umath is deprecated:DeprecationWarning"
    )
    
    # Filter out warnings from faiss.loader
    config.addinivalue_line(
        "filterwarnings",
        "ignore::DeprecationWarning:faiss.loader"
    ) 