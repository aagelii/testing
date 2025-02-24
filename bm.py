"""BMDL scoring implementation using new paradigm"""
import numpy as np
from canary import calculation_parameters as calc_p
from canary import scored_model_parameters as smp
from canary import time_series as ts
from canary.estimator_and_scorer import main, scorer
from canary.numpy_supplement import date, ndarray_extensions as ne

class BMDLScoringMethod:
    """Implements BMDL scoring using Canary's internal components"""
    def __init__(self, params_canary=None):
        self.params = params_canary or {}
        self._setup_parameters()
        self.raw_scores = []
        
    def _setup_parameters(self):
        """Initialize parameters from Canary configuration"""
        self.period = self._get_period(self.params.get("period", 12))
        self.nu = self.params.get("nu", 5)
        self.alpha = self.params.get("alpha", 1)
        self.b_eta = self.params.get("b_eta", 60)
        self.beta_supposed = self.params.get("beta_supposed", 
                                           self.b_eta / scorer.beta_binomial_parameters.BETA_SUPPOSED_DIVISOR)
        
        self.lag1 = self._get_lag1()
        self.meta = self._get_meta()
        self.weights = self._get_weights()
        
    def model(self, segment):
        """Construct time series model parameters for a segment"""
        return tsmp.TimeSeriesModelParameters(
            ne.IndicatorVector(np.zeros(len(segment), dtype=int)),  # No internal changepoints
            self.params.get("sin_cos_pairs", 0),
            amp.AutoregressiveModelParameters(order=self.params.get("ar_order", 1))
        )

    def loss(self, segment):
        """Calculate BMDL loss for a segment"""
        time_series = ts.TimeSeries(
            observations=ne.Vector(segment),
            timestamps=date.Sequence(np.arange(len(segment)), self.period)
        )
        
        calculation_params = calc_p.CalculationParameters(
            time_series=time_series,
            variance_multiplier=self.nu,
            scorer_configuration=scorer.full.Configuration(
                supposed_changepoints=self.meta,
                a_lag_1_ac_scorer=self.lag1,
                beta_binomial_parameters=scorer.beta_binomial_parameters.BetaBinomialParameters(
                    alpha=self.alpha, beta=self.b_eta, beta_supposed=self.beta_supposed
                )
            ),
            precision=1e-8,
            weights=self.weights
        )
        
        main_builder = main.EstimatorAndScorer(self.model(segment), calculation_params)
        return main_builder.estimate_and_score().scored_parameters.bmdl_score

    def compare(self, loss_with, loss_without):
        """Calculate BMDL improvement from changepoint"""
        return loss_without - loss_with

    def normalize(self, score):
        """Apply Canary's standard normalization"""
        clipped = np.clip(score, 1, None)
        return np.minimum(np.round(np.log(clipped) * 10), 100)

    # Helper methods from original implementation
    def _get_period(self, period_raw):
        # Original _get_period implementation
        if period_raw == 12: return date.Period.MONTHS_PER_YEAR
        elif period_raw == 52: return date.Period.FLOOR_OF_WEEKS_PER_YEAR
        elif period_raw == 53: return date.Period.CEILING_OF_WEEKS_PER_YEAR
        elif period_raw == 7: return date.Period.DAYS_PER_WEEK
        elif period_raw == 24: return date.Period.HOURS_PER_DAY
        raise ValueError(f"Unsupported period: {period_raw}")

    def _get_lag1(self):
        # Original _get_lag1 logic
        lag1_type = self.params.get("r1_prior", "t-norm")
        if lag1_type == "t-norm":
            return scorer.lag_1_ac.TanTransformedNormal(
                standard_deviation=self.params.get("std_r1", 0.35)
            )
        elif lag1_type == "beta":
            return scorer.lag_1_ac.Beta(
                beta_parameter=self.params.get("a_r1", 1)
            )
        raise ValueError(f"Unsupported lag1 type: {lag1_type}")

    def _get_meta(self):
        # Original _get_meta logic
        meta_raw = self.params.get("meta")
        if meta_raw is None: 
            return ne.IndicatorVector(np.zeros(0, dtype=int))
        return ne.IndicatorVector.from_indices(len(meta_raw), meta_raw)

    def _get_weights(self):
        # Original _get_weights logic
        weights_raw = self.params.get("weights")
        return ws.Weights(np.ones_like(weights_raw)) if weights_raw is None else ws.Weights(weights_raw)

    def compute_scores(self, full_series, changepoints):
        """Main scoring workflow matching original calculate_severity"""
        scores = np.full(len(full_series), np.nan)
        
        for cp in np.where(changepoints)[0]:
            prev_cp = np.where(changepoints[:cp])[0][-1] if np.any(changepoints[:cp]) else 0
            next_cp = np.where(changepoints[cp:])[0][0]+cp if np.any(changepoints[cp:]) else len(full_series)
            
            G1 = full_series[prev_cp:cp]
            G2 = full_series[cp:next_cp]
            merged = np.concatenate([G1, G2])
            
            loss_with = self.loss(G1) + self.loss(G2)
            loss_without = self.loss(merged)
            
            scores[cp] = self.compare(loss_with, loss_without)
            self.raw_scores.append(scores[cp])
            
        return self.normalize(scores)
