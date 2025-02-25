import numpy as np
from canary import calculation_parameters as calc_p
from canary import scored_model_parameters as smp
from canary import time_series as ts
from canary.additive_decomposition import Builder as ad_builder
from canary.estimator_and_scorer import autoregressive_model_parameters as amp
from canary.estimator_and_scorer import main, scorer
from canary.estimator_and_scorer import time_series_model_parameters as tsmp
from canary.estimator_and_scorer import weights as ws
from canary.estimator_and_scorer.scorer import beta_binomial_parameters as bbp
from canary.estimator_and_scorer.scorer import lag_1_ac as lag_1_ac_scorer
from canary.numpy_supplement import date
from canary.numpy_supplement import ndarray_extensions as ne
from canary.regime_parameters import Builder as rp_builder
from canary.report import Report

class BMDLScoringMethod(ScoringMethod):
    def __init__(self, period=12, nu=5, a=1, b_eta=60, b_eta_meta=None, r1_prior="t-norm", a_r1=1, std_r1=0.35, meta=None, weights=None, handle_neg_severity=False, normalize_severity_scores=False, **kwargs):
        super().__init__()
        if b_eta_meta is None:
            b_eta_meta = b_eta / bbp.BETA_SUPPOSED_DIVISOR
        self.params_canary = {
            "period": period,
            "nu": nu,
            "alpha": a,
            "b_eta": b_eta,
            "beta_supposed": b_eta_meta,
            "r1_prior": r1_prior,
            "a_r1": a_r1,
            "std_r1": std_r1,
            "meta": meta,
            "weights": weights,
        }
        self.handle_neg_severity = handle_neg_severity
        self.normalize_severity_scores = normalize_severity_scores
        self.original_model_params = {}
        self.original_loss = None

    def model(self, segment):
        self.original_model_params = {
            'observations': segment.observations,
            'dates': segment.dates,
            'changepoints': segment.changepoints,
            'ar_order': segment.ar_order,
            'sin_cos_pairs': segment.sin_cos_pairs,
        }
        self.original_loss = self.loss(**self.original_model_params)

    def loss(self, observations, dates, changepoints, ar_order, sin_cos_pairs):
        period_raw = self.params_canary.get("period", 12)
        period = self._get_period(period_raw)
        time_series = ts.TimeSeries(
            observations=ne.Vector(observations),
            timestamps=date.Sequence(sequence=dates, period=period),
        )
        input_ar_model_parameters = amp.AutoregressiveModelParameters(order=ar_order)
        dates_clean = dates[~np.isnan(changepoints)]
        changepoints_clean = changepoints[~np.isnan(changepoints)]
        model_parameters = tsmp.TimeSeriesModelParameters(
            ne.IndicatorVector(changepoints_clean.astype(int)),
            sin_cos_pairs,
            input_ar_model_parameters,
        )
        lag1 = self._get_lag1(self.params_canary.get("r1_prior"), self.params_canary)
        meta = self._get_meta(self.params_canary.get("meta"), dates)
        weights = self._get_weights(self.params_canary.get("weights"), dates)
        calculation_parameters = calc_p.CalculationParameters(
            time_series=time_series,
            variance_multiplier=self.params_canary.get("nu", 5),
            scorer_configuration=scorer.full.Configuration(
                supposed_changepoints=meta,
                a_lag_1_ac_scorer=lag1,
                beta_binomial_parameters=bbp.BetaBinomialParameters(
                    alpha=self.params_canary.get("alpha", 1),
                    beta=self.params_canary.get("b_eta", 60),
                    beta_supposed=self.params_canary.get("beta_supposed"),
                ),
            ),
            precision=1e-8,
            weights=weights,
        )
        main_builder = main.EstimatorAndScorer(model_parameters, calculation_parameters)
        score_parameters_and_ir = main_builder.estimate_and_score()
        return score_parameters_and_ir.scored_parameters.bmdl_score

    def compare(self):
        cps = np.where(self.original_model_params['changepoints'] == 1)[0]
        severity_scores = np.full_like(self.original_model_params['changepoints'], np.nan, dtype=float)

        for cp in cps:
            modified_changepoints = self.original_model_params['changepoints'].copy()
            modified_changepoints[cp] = 0
            modified_loss = self.loss(
                observations=self.original_model_params['observations'],
                dates=self.original_model_params['dates'],
                changepoints=modified_changepoints,
                ar_order=self.original_model_params['ar_order'],
                sin_cos_pairs=self.original_model_params['sin_cos_pairs'],
            )
            severity_scores[cp] = modified_loss - self.original_loss

        return severity_scores

    def normalize(self, severity_scores):
        scores = severity_scores.copy()
        if self.handle_neg_severity:
            min_raw = np.nanmin(scores)
            if min_raw < 0:
                scores += abs(min_raw)
        if self.normalize_severity_scores:
            scores = np.maximum(scores, 1)
            scores = np.log(scores)
            scores = np.round(scores * 10)
            scores = np.minimum(scores, 100)
            scores = np.maximum(scores, 1)
        return scores

    def _get_period(self, period_raw):
        if period_raw == 12:
            return date.Period.MONTHS_PER_YEAR
        elif period_raw == 52:
            return date.Period.FLOOR_OF_WEEKS_PER_YEAR
        elif period_raw == 53:
            return date.Period.CEILING_OF_WEEKS_PER_YEAR
        elif period_raw == 7:
            return date.Period.DAYS_PER_WEEK
        elif period_raw == 24:
            return date.Period.HOURS_PER_DAY
        else:
            raise ValueError(f"Period {period_raw} unsupported, must be from [7, 12, 24, 52, 53]")

    def _get_lag1(self, lag1_raw, params_canary):
        if lag1_raw == "t-norm":
            return lag_1_ac_scorer.TanTransformedNormal(
                standard_deviation=params_canary.get("std_r1", 0.35)
            )
        elif lag1_raw == "beta":
            return lag_1_ac_scorer.Beta(
                beta_parameter=params_canary.get("a_r1", 1)
            )
        else:
            raise ValueError(f"Lag1 {lag1_raw} unsupported! Must be from [t-norm, beta]")

    def _get_weights(self, weights_raw, dates):
        if weights_raw is None:
            return ws.Weights(np.ones(len(dates), dtype=int)
        else:
            return ws.Weights(np.array(weights_raw))

    def _get_meta(self, meta_raw, dates):
        if meta_raw is None:
            return ne.IndicatorVector(np.zeros(len(dates), dtype=int)
        elif len(meta_raw) < len(dates):
            return ne.IndicatorVector.from_indices(len(dates), meta_raw)
        else:
            return ne.IndicatorVector(np.array(meta_raw, dtype=int))
