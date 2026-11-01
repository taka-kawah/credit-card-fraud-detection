import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler

from read import read_datasets

# とりあえずamountでgaussian kernel とって裾0.1%を検知してみる

train_all, test_all = read_datasets()
train_amount = train_all["Amount"]

scaler = StandardScaler()

normalized_train = scaler.fit_transform(np.log1p(train_amount).to_frame())
kde = KernelDensity(kernel="gaussian", bandwidth="scott", rtol=1e-4).fit(
    normalized_train
)
grid = np.linspace(
    normalized_train.min() - 3, normalized_train.max() + 3, 2000
).reshape(-1, 1)
grid_score = kde.score_samples(grid)
score = np.interp(normalized_train[:, 0], grid[:, 0], grid_score)

threshold = np.quantile(score, 0.001)

test = test_all[["Amount", "Class"]]
test["Amount"] = scaler.transform(np.log1p(test["Amount"]).to_frame())
not_detected = test[test["Amount"] >= threshold]
detected = test[test["Amount"] < threshold]

misdetected = detected[detected["Class"] == 0]
failed_to_detect = not_detected[not_detected["Class"] == 1]
print(f"detected: {len(detected)}")
print(f"misdetected: {len(misdetected)}")
print(f"failed to detect: {len(failed_to_detect)}")

"""
credit-card-fraud-detection on  main [?] via  v3.15.1 (.venv)
❯ duckdb 3/creditcard.csv "SELECT COUNT(*) FROM (SELECT * ,PERCENT_RANK() OVER (ORDER BY Time ASC) AS p_rank FROM creditcard) WHERE Class=1 AND p_rank >= 0.8"
┌──────────────┐
│ count_star() │
│    int64     │
├──────────────┤
│           75 │
└──────────────┘

credit-card-fraud-detection on  main [?] via  v3.15.1 (.venv)
❯ python kde_amount.py
detected: 0
misdetected: 0
failed to detect: 75

全部逃してて草
"""
