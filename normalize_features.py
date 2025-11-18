# normalize_features.py
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# ==== CONFIG: set your CSV and column lists ====
CSV_PATH = "daily_features.csv"   # first column must be the date
OUT_CSV  = "daily_features_normalized.csv"
PREPROC_PATH = "normalizer.joblib"
TRAIN_FRAC = 0.80  # fit scalers on first 80% of rows (chronological)

# Choose exactly one bucket per feature (exclude the date column)
LEAVE_AS_IS = [
    "tweet_ratio", "avg_tweets_per_hour", "burstiness",
    "time_of_first_tweet_sin","time_of_first_tweet_cos",
    "time_of_last_tweet_sin","time_of_last_tweet_cos",
]
STANDARDIZE = [
    "avg_tweet_length","tweet_entropy",
]
LOGSTD_COUNTS = [
    "tweets_total","retweets_total","active_hours_count",
    "morning_tweets","afternoon_tweets","evening_tweets","night_tweets",
    "burst_count","max_burst_length","7d_avg_tweets","7d_var_tweets",
]
LOGSTD_GAPS = [
    "mean_intertweet_gap","std_intertweet_gap","min_gap","max_gap",
]

# ==== Build preprocessor ====
log1p_std = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("std", StandardScaler())
])
std = Pipeline([("std", StandardScaler())])

def main():
    df = pd.read_csv(CSV_PATH, parse_dates=[0])
    df = df.sort_values(df.columns[0]).reset_index(drop=True)
    date_col = df.columns[0]

    # Coerce non-date columns to numeric where possible
    for c in df.columns[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Sanity: check membership and warn for unassigned columns
    all_feats = set(df.columns[1:])
    assigned = set(LEAVE_AS_IS) | set(STANDARDIZE) | set(LOGSTD_COUNTS) | set(LOGSTD_GAPS)
    missing = all_feats - assigned
    if missing:
        print("WARNING: Some numeric columns are not assigned to any normalization bucket and will be DROPPED:")
        print(sorted(missing))

    # ColumnTransformer
    ct = ColumnTransformer(
        transformers=[
            ("leave", "passthrough", [c for c in LEAVE_AS_IS if c in df.columns]),
            ("std",   std,           [c for c in STANDARDIZE if c in df.columns]),
            ("logstd_counts", log1p_std, [c for c in LOGSTD_COUNTS if c in df.columns]),
            ("logstd_gaps",   log1p_std, [c for c in LOGSTD_GAPS if c in df.columns]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    # Fit on first TRAIN_FRAC of rows (chronological)
    n = len(df)
    split = max(1, int(n * TRAIN_FRAC))
    X_train = df.iloc[:split, 1:]  # exclude date col
    X_all   = df.iloc[:, 1:]

    ct.fit(X_train)

    X_scaled = ct.transform(X_all)
    feat_names = ct.get_feature_names_out()
    df_scaled = pd.DataFrame(X_scaled, columns=feat_names, index=df.index)

    # Keep date column as-is (first column)
    out = pd.concat([df[[date_col]].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
    out.to_csv(OUT_CSV, index=False)
    joblib.dump(ct, PREPROC_PATH)

    print(f"Fitted on first {split}/{n} rows (TRAIN_FRAC={TRAIN_FRAC:.2f}).")
    print(f"Wrote normalized CSV -> {Path(OUT_CSV).resolve()}")
    print(f"Saved preprocessor -> {Path(PREPROC_PATH).resolve()}")

if __name__ == "__main__":
    main()
