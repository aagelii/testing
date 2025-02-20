Here's a comprehensive pytest test suite for your changepoint scoring system. Create a file called `test_scoring.py`:

```python
import pytest
import numpy as np
from scoring import (
    ScoringMethod,
    BMDLScoringMethod,
    TripleExpScoringMethod,
    OriginalScoringMethod
)

@pytest.fixture
def sample_segments():
    """Generate test segments with known properties"""
    return {
        'constant': np.ones(50),
        'step_change': np.concatenate([np.ones(25), np.ones(25)+5]),
        'trend_change': np.concatenate([np.linspace(0, 5, 25), np.linspace(5, 0, 25)]),
        'noisy': np.random.normal(0, 1, 50)
    }

class TestBaseScoring:
    @pytest.fixture
    def scorer(self):
        """Abstract base class testing requires concrete implementation"""
        return TripleExpScoringMethod()

    def test_abstract_methods(self):
        """Ensure base class can't be instantiated directly"""
        with pytest.raises(TypeError):
            ScoringMethod()

    def test_compute_score_interface(self, scorer, sample_segments):
        """Test basic scoring workflow"""
        G1 = sample_segments['step_change'][:25]
        G2 = sample_segments['step_change'][25:]
        
        raw_score = scorer.compute_score(G1, G2)
        scorer.finalize()
        normalized = scorer.get_normalized_scores()
        
        assert isinstance(raw_score, float)
        assert len(normalized) == 1
        assert 0 <= normalized[0] <= 100

class TestBMDLScoring:
    @pytest.fixture
    def scorer(self):
        return BMDLScoringMethod(mcmc_samples=100, tune=50)  # Faster for testing

    def test_bmdl_loss_calculation(self, scorer):
        """Test BMDL loss decreases with better fits"""
        good_fit = np.random.normal(0, 1, 100)
        bad_fit = np.random.uniform(-10, 10, 100)
        
        loss_good = scorer.loss(good_fit)
        loss_bad = scorer.loss(bad_fit)
        
        assert loss_good < loss_bad
        assert isinstance(loss_good, float)
        assert isinstance(loss_bad, float)

    def test_score_direction(self, scorer, sample_segments):
        """Test that true changepoints get higher scores"""
        true_cp = sample_segments['step_change']
        G1_true = true_cp[:25]
        G2_true = true_cp[25:]
        
        false_cp = sample_segments['constant']
        G1_false = false_cp[:25]
        G2_false = false_cp[25:]
        
        score_true = scorer.compute_score(G1_true, G2_true)
        score_false = scorer.compute_score(G1_false, G2_false)
        
        assert score_true > score_false

class TestTripleExpScoring:
    @pytest.fixture
    def scorer(self):
        return TripleExpScoringMethod()

    def test_model_fitting(self, scorer):
        """Test model can fit different trend types"""
        ascending = np.linspace(0, 10, 50)
        descending = np.linspace(10, 0, 50)
        
        model_asc = scorer.model(ascending)
        model_desc = scorer.model(descending)
        
        assert model_asc.params['trend'] > 0
        assert model_desc.params['trend'] < 0

    def test_edge_cases(self, scorer):
        """Test handling of short/empty segments"""
        with pytest.warns(UserWarning):
            assert scorer.loss(np.array([1])) == np.inf
            
        assert scorer.loss(np.array([])) == np.inf

class TestOriginalScoring:
    @pytest.fixture
    def scorer(self):
        return OriginalScoringMethod()

    def test_component_calculation(self, scorer, sample_segments):
        """Test mean/std/slope difference calculations"""
        G1 = sample_segments['step_change'][:25]
        G2 = sample_segments['step_change'][25:]
        
        diff = scorer.compare(scorer.loss(G1), scorer.loss(G2))
        assert len(diff) == 3  # (mean_diff, std_diff, slope_diff)
        assert diff[0] > 4.9  # Expected mean difference ~5

    def test_normalization(self, scorer, sample_segments):
        """Test global normalization across all changepoints"""
        segments = [
            (sample_segments['step_change'][:25], sample_segments['step_change'][25:]),
            (sample_segments['trend_change'][:25], sample_segments['trend_change'][25:])
        ]
        
        for G1, G2 in segments:
            scorer.compute_score(G1, G2)
            
        normalized = scorer.compute_all_scores()
        assert all(25 <= score <= 100 for score in normalized)
        assert len(normalized) == 2

def test_levenshtein_ranking():
    """Test ranking distance metric"""
    from utils import levenshtein_rank_distance
    
    perfect = [1, 2, 3]
    good = [2, 1, 3]
    bad = [3, 2, 1]
    
    assert levenshtein_rank_distance(perfect, perfect) == 0
    assert levenshtein_rank_distance(perfect, good) == 2  # Swap first two
    assert levenshtein_rank_distance(perfect, bad) == 4   # Reverse requires 4 edits
```

### Key Features:
1. **Fixtures for Test Data**: Reusable segment patterns
2. **Base Class Validation**: Ensures interface compliance
3. **Method-Specific Tests**: 
   - BMDL: Loss calculation sanity checks
   - Triple Exp: Model fitting verification
   - Original: Component calculations
4. **Edge Case Handling**: Short/empty segments
5. **Metric Validation**: Ranking distance tests

### Running Tests:
```bash
# Run all tests
pytest -v test_scoring.py

# Run specific test class
pytest -v test_scoring.py::TestBMDLScoring

# Generate coverage report
pytest --cov=scoring --cov-report=html
```

### Recommended Additions:
1. **Benchmark Tests**: Timing critical functions
```python
def test_bmdl_performance(benchmark, scorer, sample_segments):
    G1 = sample_segments['step_change'][:25]
    G2 = sample_segments['step_change'][25:]
    benchmark(scorer.compute_score, G1, G2)
```

2. **Property-Based Testing**:
```python
from hypothesis import given, strategies as st

@given(st.lists(st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
def test_normalization_bounds(scorer, data):
    scorer.raw_scores = data
    scorer.finalize()
    normalized = scorer.get_normalized_scores()
    assert all(0 <= s <= 100 for s in normalized)
```

3. **Visual Regression Tests** (if using plotting):
```python
def test_plot_output(scorer, sample_segments):
    fig = plot_changepoints(...)
    pytest.mpl.compare_axes(fig.axes, 'plot_baseline.png')
```
