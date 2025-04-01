"""
Test runner script for the OsireRAG project.

This script runs all unit tests in the tests directory.

Author: Adam Haile
Date: 3/31/2024
"""

import pytest
import sys
import os
import warnings
from collections import defaultdict
from colorama import init, Fore, Style

# Initialize colorama
init()

def run_tests():
    """Run all tests in the tests directory, excluding this runner script."""
    # Get the absolute path to the tests directory
    tests_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Filter out the known FAISS-related numpy warning
    warnings.filterwarnings("ignore", message="numpy.core._multiarray_umath is deprecated")
    
    # Run pytest with verbose output, excluding this runner script
    args = [
        "-xvs",  # Exit on first failure, verbose, no capture
        "--tb=short",  # Short traceback format
        "--color=yes",  # Colored output
        # Filter warnings
        "-W", "ignore::DeprecationWarning:faiss.loader",  # Ignore DeprecationWarning from faiss.loader
    ]
    
    # Collect test files
    test_files = []
    for file in os.listdir(tests_dir):
        if file.startswith("test_") and file.endswith(".py"):
            test_files.append(os.path.join(tests_dir, file))
            args.append(os.path.join(tests_dir, file))
    
    # Print summary of test files
    print(f"\n{Fore.CYAN}Running tests for OsireRAG:{Style.RESET_ALL}")
    for i, file in enumerate(test_files, 1):
        file_name = os.path.basename(file)
        print(f"{Fore.CYAN}{i}. {file_name}{Style.RESET_ALL}")
    print("")
    
    # Run the tests
    result = pytest.main(args)
    
    # Print summary
    print(f"\n{Fore.GREEN}Test execution complete with exit code: {result}{Style.RESET_ALL}")
    
    return result

if __name__ == "__main__":
    sys.exit(run_tests()) 