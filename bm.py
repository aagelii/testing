"""Severity scoring with generic scoring framework"""
from typing import Optional, Callable, Dict
import numpy as np
from canary import calculation_parameters as calc_p
from canary import scored_model_parameters as smp
from canary import time_series as ts
from canary.numpy_supplement import date
from canary.numpy_supplement import ndarray_extensions as ne

class GenericScoringReport:
    """Container for generic scoring results matching Canary's interface"""
    def __init__(self, time_series, scores, normalized_scores, model_params):
        self.input_time_series = time_series
        self.scored_parameters = smp.ScoredModelParameters(
            parameters=model_params,
            bmdl_score=np.nan,  # Maintain interface compatibility
            severity_score=normalized_scores
        )
        self.raw_scores = scores

class GenericScorer:
    """Core scoring class mirroring Canary's BMDL structure"""
    def __init__(
        self,
        model_func: Callable,
        loss_func: Callable, 
        compare_func: Callable,
        norm_func: Callable,
        params: Dict
    ):
        self.model = model_func
        self.loss = loss_func
        self.compare = compare_func
        self.normalize = norm_func
        self.params = params
        self.raw_scores = []
        self.normalized_scores = []

    def _prepare_time_series(self, observations, dates):
        """Mirror Canary's time series preparation"""
        period = self.params.get("period", 12)
        return ts.TimeSeries(
            observations=ne.Vector(observations),
            timestamps=date.Sequence(sequence=dates, period=period)
        )

    def _compute_segment_score(self, G1, G2):
        """Core scoring logic matching your framework"""
        # With changepoint
        model_G1 = self.model(G1)
        model_G2 = self.model(G2)
        loss_with = self.loss(model_G1, G1) + self.loss(model_G2, G2)

        # Without changepoint
        merged = np.concatenate([G1, G2])
        model_merged = self.model(merged)
        loss_without = self.loss(model_merged, merged)

        return self.compare(loss_with, loss_without)

    def calculate_scores(self, observations, dates, changepoints):
        """Main scoring workflow matching Canary's calc_bmdl structure"""
        time_series = self._prepare_time_series(observations, dates)
        cps = np.where(changepoints == 1)[0]

        # Compute scores for each changepoint
        for cp in cps:
            prev_cp = np.where(changepoints[:cp] == 1)[0][-1] if np.any(changepoints[:cp]) else 0
            next_cp = np.where(changepoints[cp:] == 1)[0][0] + cp if np.any(changepoints[cp:]) else len(observations)
            
            G1 = observations[prev_cp:cp]
            G2 = observations[cp:next_cp]
            
            score = self._compute_segment_score(G1, G2)
            self.raw_scores.append((cp, score))

        # Normalization
        scores = np.full_like(observations, np.nan, dtype=float)
        for cp, score in self.raw_scores:
            scores[cp] = score
            
        self.normalized_scores = self.normalize(scores)
        
        return GenericScoringReport(
            time_series=time_series,
            scores=scores,
            normalized_scores=self.normalized_scores,
            model_params=self.params
        )

def generic_calc_score(
    observations: np.ndarray,
    dates: np.ndarray,
    changepoints: np.ndarray,
    model_func: Callable,
    loss_func: Callable,
    compare_func: Callable,
    norm_func: Callable,
    params: Optional[dict] = None
) -> GenericScoringReport:
    """Entry point matching Canary's calc_bmdl signature"""
    if params is None:
        params = {}
        
    scorer = GenericScorer(
        model_func=model_func,
        loss_func=loss_func,
        compare_func=compare_func,
        norm_func=norm_func,
        params=params
    )
    
    return scorer.calculate_scores(observations, dates, changepoints)

# Example usage matching Canary's pattern:
def triple_exp_model(segment):
    # Implement your model fitting here
    pass

def mse_loss(model, segment):
    # Implement MSE calculation
    pass

def score_compare(loss_with, loss_without):
    return loss_without - loss_with

def minmax_normalize(scores):
    # Implement normalization
    pass

report = generic_calc_score(
    observations=data.observations,
    dates=data.timestamps,
    changepoints=results.best.eta,
    model_func=triple_exp_model,
    loss_func=mse_loss,
    compare_func=score_compare,
    norm_func=minmax_normalize,
    params={"period": 12}
)
