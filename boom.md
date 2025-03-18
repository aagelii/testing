
To address the challenges of scoring change points post-detection, we propose a robust methodology that combines multiple normalized features and incorporates segment length weighting. This approach aims to provide a generalized, quantitative measure of change point significance across different time series.

### Methodology
1. **Feature Extraction**: For each change point, compute differences in various statistical properties between the segments before and after the change.
2. **Normalization**: Normalize each feature by its Median Absolute Deviation (MAD) across all detected change points to ensure comparability.
3. **Segment Length Weighting**: Adjust scores by the harmonic mean of segment lengths to penalize short segments.
4. **Score Aggregation**: Sum the normalized and weighted features to produce a final score for each change point.

### Solution Code
```python
import numpy as np
from scipy.stats import ks_2samp, wasserstein_distance, linregress
from scipy.signal import detrend
from antropy import app_entropy  # Install using: pip install antropy

def compute_features(left_segment, right_segment):
    features = {}
    
    # Mean difference
    mean_left = np.mean(left_segment)
    mean_right = np.mean(right_segment)
    features['mean_diff'] = np.abs(mean_right - mean_left)
    
    # Std difference
    std_left = np.std(left_segment, ddof=1)
    std_right = np.std(right_segment, ddof=1)
    features['std_diff'] = np.abs(std_right - std_left)
    
    # Slope difference
    slope_left, slope_right = 0, 0
    if len(left_segment) >= 2:
        x_left = np.arange(len(left_segment))
        slope_left, _, _, _, _ = linregress(x_left, left_segment)
    if len(right_segment) >= 2:
        x_right = np.arange(len(right_segment))
        slope_right, _, _, _, _ = linregress(x_right, right_segment)
    features['slope_diff'] = np.abs(slope_right - slope_left)
    
    # KS statistic
    if len(left_segment) > 0 and len(right_segment) > 0:
        ks_stat, _ = ks_2samp(left_segment, right_segment)
        features['ks_stat'] = ks_stat
    else:
        features['ks_stat'] = 0.0
    
    # Wasserstein distance
    if len(left_segment) > 0 and len(right_segment) > 0:
        wd = wasserstein_distance(left_segment, right_segment)
        features['wasserstein'] = wd
    else:
        features['wasserstein'] = 0.0
    
    # Autocorrelation difference at lag 1
    def autocorr(series, lag=1):
        if len(series) < lag + 1:
            return 0.0
        return np.corrcoef(series[:-lag], series[lag:])[0, 1]
    ac_left = autocorr(left_segment, 1)
    ac_right = autocorr(right_segment, 1)
    features['ac_diff'] = np.abs(ac_right - ac_left)
    
    # Approximate entropy difference
    def compute_entropy(series):
        if len(series) < 3:
            return 0.0
        return app_entropy(series, order=2, approximate=True)
    entropy_left = compute_entropy(left_segment)
    entropy_right = compute_entropy(right_segment)
    features['entropy_diff'] = np.abs(entropy_right - entropy_left)
    
    return features

def score_change_points(ts, change_points):
    if len(change_points) == 0:
        return []
    
    features_list = []
    for cp in change_points:
        left = ts[:cp]
        right = ts[cp:]
        features = compute_features(left, right)
        features_list.append(features)
    
    feature_names = features_list[0].keys()
    feature_dict = {fn: [] for fn in feature_names}
    for f in features_list:
        for fn in feature_names:
            feature_dict[fn].append(f[fn])
    
    # Compute MAD for each feature
    mad_dict = {}
    for fn in feature_names:
        data = np.array(feature_dict[fn])
        median = np.median(data)
        mad = np.median(np.abs(data - median))
        mad_dict[fn] = mad + 1e-9  # Avoid division by zero
    
    # Compute segment length weights
    weights = []
    for cp in change_points:
        n_left = cp
        n_right = len(ts) - cp
        if n_left == 0 or n_right == 0:
            weight = 0.0
        else:
            weight = np.sqrt(n_left * n_right) / (n_left + n_right)
        weights.append(weight)
    weights = np.array(weights)
    
    # Normalize features and compute scores
    scores = []
    for i in range(len(change_points)):
        score = 0.0
        for fn in feature_names:
            value = feature_dict[fn][i]
            normalized = value / mad_dict[fn]
            score += normalized * weights[i]
        scores.append(score)
    
    return scores
```

### Explanation
1. **Feature Extraction**: The `compute_features` function calculates differences in mean, standard deviation, slope, Kolmogorov-Smirnov statistic, Wasserstein distance, autocorrelation, and entropy between segments.
2. **Robust Normalization**: Each feature is normalized using MAD to account for variability within the time series, ensuring features are scaled appropriately.
3. **Segment Length Weighting**: Scores are adjusted based on segment lengths to reduce the impact of short segments, using the harmonic mean of the lengths.
4. **Aggregation**: The final score is a weighted sum of normalized features, providing a comprehensive measure of change point significance.

This approach addresses the outlined challenges by combining multiple robust metrics, normalizing them appropriately, and penalizing unreliable short segments, resulting in a more flexible and generalizable scoring system.
