"""
Unit tests for the ReciprocalRankFusion class.

This module contains test cases for the ReciprocalRankFusion class, testing its initialization,
and the ranks method for combining document rankings.

Author: Adam Haile
Date: 3/31/2024
"""

import pytest
from unittest.mock import patch, MagicMock
from app.core.models.rrf import ReciprocalRankFusion
from app.core.models.chunker import Chunk

@pytest.fixture
def test_chunks():
    """Create test chunks for testing."""
    return [
        Chunk(content=f"Test document {i}", index=i, metadata={"id": f"doc{i}"})
        for i in range(5)
    ]

@pytest.fixture
def test_rankings(test_chunks):
    """Create test rankings for different rankers."""
    # First ranker - ranks chunks 0, 1, 2 with scores
    rank1 = [
        (test_chunks[0], 0.9),
        (test_chunks[1], 0.7),
        (test_chunks[2], 0.5)
    ]
    
    # Second ranker - ranks chunks 1, 2, 3 with scores
    rank2 = [
        (test_chunks[1], 0.8),
        (test_chunks[2], 0.6),
        (test_chunks[3], 0.4)
    ]
    
    # Third ranker - ranks chunks 2, 3, 4 with scores
    rank3 = [
        (test_chunks[2], 0.95),
        (test_chunks[3], 0.75),
        (test_chunks[4], 0.45)
    ]
    
    return [rank1, rank2, rank3]

def test_rrf_initialization():
    """Test ReciprocalRankFusion initialization."""
    rrf = ReciprocalRankFusion()
    assert isinstance(rrf, ReciprocalRankFusion)

def test_rrf_with_single_ranking(test_rankings):
    """Test RRF with a single ranking."""
    rrf = ReciprocalRankFusion()
    
    # Use only the first ranking
    result = rrf.ranks(ranks=[test_rankings[0]], k=60, n=3)
    
    # Should preserve the original order but normalize scores
    assert len(result) == 3
    assert result[0][0].index == 0  # First chunk should be rank 0
    assert result[1][0].index == 1  # Second chunk should be rank 1
    assert result[2][0].index == 2  # Third chunk should be rank 2
    
    # All scores should be normalized between 0 and 1
    assert all(0 <= score <= 1 for _, score in result)
    assert result[0][1] > result[1][1] > result[2][1]  # Scores should decrease

def test_rrf_with_multiple_rankings(test_rankings):
    """Test RRF with multiple rankings."""
    rrf = ReciprocalRankFusion()
    
    # Use all rankings
    result = rrf.ranks(ranks=test_rankings, k=60, n=5)
    
    # Should have 5 results
    assert len(result) == 5
    
    # Chunk 2 appears in all rankings, so should be ranked high
    # Check that the chunks are correctly ordered by their RRF scores
    chunk_indices = [chunk.index for chunk, _ in result]
    
    # Chunk 2 should be in the top results since it appears in all rankings
    assert 2 in chunk_indices[:2]
    
    # All scores should be normalized between 0 and 1
    assert all(0 <= score <= 1 for _, score in result)
    
    # First result should have highest score (1.0)
    assert result[0][1] == 1.0

def test_rrf_with_empty_rankings():
    """Test RRF with empty rankings."""
    rrf = ReciprocalRankFusion()
    
    # Empty list of rankings
    result = rrf.ranks(ranks=[], k=60, n=5)
    
    # Should return an empty result
    assert result == []

def test_rrf_with_k_parameter(test_rankings):
    """Test RRF with different k parameters."""
    rrf = ReciprocalRankFusion()
    
    # With k=1 (smaller k makes position more important)
    result_small_k = rrf.ranks(ranks=test_rankings, k=1, n=5)
    
    # With k=100 (larger k makes position less important)
    result_large_k = rrf.ranks(ranks=test_rankings, k=100, n=5)
    
    # Both should return 5 results
    assert len(result_small_k) == 5
    assert len(result_large_k) == 5
    
    # Results order might be different due to k parameter
    # But both should have valid normalized scores
    assert all(0 <= score <= 1 for _, score in result_small_k)
    assert all(0 <= score <= 1 for _, score in result_large_k)

def test_rrf_with_n_parameter(test_rankings):
    """Test RRF with different n parameters."""
    rrf = ReciprocalRankFusion()
    
    # Get top 2 results
    result_n2 = rrf.ranks(ranks=test_rankings, k=60, n=2)
    
    # Get top 4 results
    result_n4 = rrf.ranks(ranks=test_rankings, k=60, n=4)
    
    # Should respect the n parameter
    assert len(result_n2) == 2
    assert len(result_n4) == 4
    
    # The top 2 results should match between both results
    assert [chunk.index for chunk, _ in result_n2] == [chunk.index for chunk, _ in result_n4[:2]]

def test_rrf_with_threshold(test_rankings):
    """Test RRF with threshold parameter."""
    rrf = ReciprocalRankFusion()
    
    # Set a high threshold that should filter out some results
    result = rrf.ranks(ranks=test_rankings, k=60, n=5, threshold=0.8)
    
    # All returned scores should be >= threshold
    assert all(score >= 0.8 for _, score in result)
    
    # When threshold is too high, might return fewer results than n
    assert len(result) <= 5

def test_rrf_with_duplicate_chunks(test_chunks):
    """Test RRF with duplicate chunks in rankings."""
    # Create rankings with duplicated chunks
    rank1 = [(test_chunks[0], 0.9), (test_chunks[1], 0.7)]
    rank2 = [(test_chunks[0], 0.8), (test_chunks[2], 0.6)]  # Duplicate chunk 0
    
    rrf = ReciprocalRankFusion()
    result = rrf.ranks(ranks=[rank1, rank2], k=60, n=3)
    
    # Should handle duplicates correctly
    assert len(result) == 3
    
    # Chunk 0 appears in both rankings, so should be ranked high
    assert result[0][0].index == 0
    
    # No duplicate chunks in the result
    chunk_indices = [chunk.index for chunk, _ in result]
    assert len(chunk_indices) == len(set(chunk_indices))

def test_rrf_logging(test_rankings):
    """Test RRF logs results correctly."""
    rrf = ReciprocalRankFusion()
    
    # Use a context manager to patch the logger
    with patch('app.core.models.rrf.logger') as mock_logger:
        result = rrf.ranks(ranks=test_rankings, k=60, n=3)
        
        # Should log the results
        mock_logger.info.assert_called_once()
        
        # The log message should contain the truncated content and scores
        log_msg = mock_logger.info.call_args[0][0]
        assert "RRF results" in log_msg
        assert "Test document" in log_msg

def test_rrf_equal_scores():
    """Test RRF when all documents have equal scores."""
    # Create chunks
    chunks = [
        Chunk(content=f"Test document {i}", index=i, metadata={})
        for i in range(3)
    ]
    
    # Create rankings with equal scores
    rank1 = [(chunks[0], 0.5), (chunks[1], 0.5)]
    rank2 = [(chunks[1], 0.5), (chunks[2], 0.5)]
    
    rrf = ReciprocalRankFusion()
    result = rrf.ranks(ranks=[rank1, rank2], k=60, n=3)
    
    # Should return 3 results with normalized scores
    assert len(result) == 3
    
    # When scores are equal, position in the ranking matters
    # Chunk 1 appears in both rankings, so should be ranked higher
    chunk_indices = [chunk.index for chunk, _ in result]
    assert 1 in chunk_indices[:1]  # Chunk 1 should be first or tied for first

def test_rrf_score_normalization():
    """Test RRF score normalization with extreme values."""
    # Create chunks
    chunks = [
        Chunk(content=f"Test document {i}", index=i, metadata={})
        for i in range(3)
    ]
    
    # Create rankings with very different scores
    rank1 = [(chunks[0], 1000.0), (chunks[1], 500.0)]
    rank2 = [(chunks[1], 100.0), (chunks[2], 10.0)]
    
    rrf = ReciprocalRankFusion()
    result = rrf.ranks(ranks=[rank1, rank2], k=60, n=3)
    
    # Should normalize scores between 0 and 1
    assert all(0 <= score <= 1 for _, score in result)
    assert result[0][1] == 1.0  # Highest score should be normalized to 1.0 