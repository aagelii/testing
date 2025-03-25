Here’s a Python implementation for change point detection with a composite score using multiple statistical tests and weighted aggregation.

⸻

✅ Step 1: Install Required Libraries

If you don’t have these libraries installed, run:

pip install numpy pandas scipy ruptures statsmodels scikit-learn



⸻

⚙️ Step 2: Python Implementation

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind, ks_2samp, levene
from ruptures import Binseg
from statsmodels.tsa.stattools import acf, adfuller, q_stat
from sklearn.preprocessing import MinMaxScaler

# -----------------------------
# Simulate a sample time series
# -----------------------------
np.random.seed(42)
n = 500
ts1 = np.random.normal(0, 1, n)  # Segment 1
ts2 = np.random.normal(2, 1, n)  # Segment 2 (mean shift)
ts = np.concatenate([ts1, ts2])

# ---------------------------------------
# Step 1: Detect Change Points (ruptures)
# ---------------------------------------
model = Binseg(model="l2").fit(ts)
change_points = model.predict(n_bkps=1)  # Detect 1 change point
print(f"Change points: {change_points}")

# ----------------------------------------------
# Step 2: Statistical Tests Around Change Points
# ----------------------------------------------
window = 50  # Window size for testing before and after the change
features = []

for cp in change_points[:-1]:  # Exclude end point
    before = ts[max(0, cp - window):cp]
    after = ts[cp:min(len(ts), cp + window)]
    
    # Mean shift (T-test)
    t_stat, t_p = ttest_ind(before, after)
    
    # Distribution shift (Kolmogorov-Smirnov test)
    ks_stat, ks_p = ks_2samp(before, after)
    
    # Variance shift (Levene test)
    lev_stat, lev_p = levene(before, after)
    
    # Autocorrelation change
    acf_before = acf(before, nlags=10)
    acf_after = acf(after, nlags=10)
    acf_diff = np.sum(np.abs(acf_before - acf_after))

    # Add the scores for the detected change point
    features.append({
        "change_point": cp,
        "t_test": t_stat,
        "ks_test": ks_stat,
        "levene_test": lev_stat,
        "acf_diff": acf_diff
    })

# Convert to DataFrame
df = pd.DataFrame(features)

# ----------------------------------------------
# Step 3: Normalize and Combine Scores
# ----------------------------------------------
# Normalize each test statistic between 0 and 1
scaler = MinMaxScaler()
normalized_features = scaler.fit_transform(df.drop("change_point", axis=1))

# Define weights for each score
weights = np.array([0.4, 0.3, 0.2, 0.1])  # T-test, KS, Levene, ACF

# Composite score formula
composite_scores = 100 * np.dot(normalized_features, weights)

# Add scores to the DataFrame
df["composite_score"] = composite_scores

# ----------------------------------------------
# Display Results
# ----------------------------------------------
print("\nChange Points with Composite Scores:")
print(df[["change_point", "composite_score"]])

# Plotting the time series and change points
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 6))
plt.plot(ts, label='Time Series')
for idx, row in df.iterrows():
    plt.axvline(row['change_point'], color='red', linestyle='--', label=f'Change Point {idx+1}')
    plt.text(row['change_point'] + 5, np.mean(ts), f"{row['composite_score']:.1f}", color='blue')
plt.title("Time Series with Change Points and Composite Scores")
plt.legend()
plt.show()



⸻

✅ Explanation:
	1.	Change point detection: Uses ruptures library with Binary Segmentation.
	2.	Statistical tests:
	•	T-test: Tests mean shift.
	•	Kolmogorov-Smirnov: Measures distributional shift.
	•	Levene’s test: Detects variance changes.
	•	ACF difference: Measures autocorrelation shift.
	3.	Composite score:
	•	Normalizes the test results.
	•	Combines them using weighted aggregation.
	4.	Visualization:
	•	Displays the time series.
	•	Marks detected change points with their composite scores.

⸻

🚀 Next Steps:
	•	You can adjust the weights dynamically based on importance.
	•	Add more statistical tests (e.g., KL divergence, entropy).
	•	Use grid search or optimization to tune the weights.
	•	Would you like help with feature selection, optimizing weights, or applying this on your own dataset?
