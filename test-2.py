"""This module contains tests for scoring methods in `scoring.py`."""
import numpy as np
import pytest
from scoring import (
    BMDLScoringMethod,
    TripleExpScoringMethod,
    OriginalScoringMethod
)

@pytest.fixture(name="sample_data")
def fixture_sample_data():
    """Generate test segments with known properties"""
    return {
        "constant": np.ones(50),
        "step_change": np.concatenate([np.ones(25), np.ones(25) + 5]),
        "trend_change": np.concatenate([np.linspace(0, 5, 25), np.linspace(5, 0, 25)]),
        "noisy": np.random.normal(0, 1, 50)
    }

def test_scoring_interface(sample_data):
    """Test basic scoring workflow across all methods"""
    scorers = [
        BMDLScoringMethod(mcmc_samples=100, tune=50),
        TripleExpScoringMethod(),
        OriginalScoringMethod()
    ]
    
    G1 = sample_data["step_change"][:25]
    G2 = sample_data["step_change"][25:]
    
    for scorer in scorers:
        raw_score = scorer.compute_score(G1, G2)
        scorer.finalize()
        normalized = scorer.get_normalized_scores()
        
        assert isinstance(raw_score, float)
        assert len(normalized) == 1
        assert 0 <= normalized[0] <= 100

def test_bmdl_loss_calculation():
    """Test BMDL loss decreases with better fits"""
    scorer = BMDLScoringMethod(mcmc_samples=100, tune=50)
    good_fit = np.random.normal(0, 1, 100)
    bad_fit = np.random.uniform(-10, 10, 100)
    
    assert scorer.loss(good_fit) < scorer.loss(bad_fit)
    assert isinstance(scorer.loss(good_fit), float)
    assert isinstance(scorer.loss(bad_fit), float)

def test_score_direction(sample_data):
    """Test true changepoints get higher scores than false positives"""
    scorer = BMDLScoringMethod(mcmc_samples=100, tune=50)
    true_seg = sample_data["step_change"]
    false_seg = sample_data["constant"]
    
    score_true = scorer.compute_score(true_seg[:25], true_seg[25:])
    score_false = scorer.compute_score(false_seg[:25], false_seg[25:])
    
    assert score_true > score_false

def test_triple_exp_fitting():
    """Test triple exp model fitting for different trend types"""
    scorer = TripleExpScoringMethod()
    ascending = np.linspace(0, 10, 50)
    descending = np.linspace(10, 0, 50)
    
    assert scorer.model(ascending).params["trend"] > 0
    assert scorer.model(descending).params["trend"] < 0

def test_edge_case_handling():
    """Test handling of short/empty segments"""
    scorer = TripleExpScoringMethod()
    
    with pytest.warns(UserWarning):
        assert scorer.loss(np.array([1])) == np.inf
        
    assert scorer.loss(np.array([])) == np.inf

def test_component_calculation(sample_data):
    """Test mean/std/slope difference calculations"""
    scorer = OriginalScoringMethod()
    G1 = sample_data["step_change"][:25]
    G2 = sample_data["step_change"][25:]
    
    diffs = scorer.compare(scorer.loss(G1), scorer.loss(G2))
    assert len(diffs) == 3  # (mean_diff, std_diff, slope_diff)
    assert diffs[0] > 4.9  # Expected mean difference ~5

def test_global_normalization(sample_data):
    """Test normalization across multiple changepoints"""
    scorer = OriginalScoringMethod()
    segments = [
        (sample_data["step_change"][:25], sample_data["step_change"][25:]),
        (sample_data["trend_change"][:25], sample_data["trend_change"][25:])
    ]
    
    for G1, G2 in segments:
        scorer.compute_score(G1, G2)
        
    normalized = scorer.compute_all_scores()
    assert all(25 <= score <= 100 for score in normalized)
    assert len(normalized) == 2

def test_negative_score_handling():
    """Test conversion of negative scores to non-negative while preserving order"""
    scorer = BMDLScoringMethod()
    raw_scores = [-5, 2, 10, -3]
    scorer.raw_scores = raw_scores
    non_neg_scores = scorer.handle_negative_scores()
    
    assert all(s >= 0 for s in non_neg_scores)
    assert np.array_equal(np.argsort(raw_scores), np.argsort(non_neg_scores))
