import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

# file paths
CSV_PATH = "daily_features.csv"  
OUT_CSV  = "daily_features_normalized.csv"
PREPROC_PATH = "normalizer.joblib"
TRAIN_FRAC = 0.80 

# different types of normalization
# choose which column is normalized in what way
LEAVE_AS_IS = [ 
    "tweet_ratio", "avg_tweets_per_hour", "burstiness",
    "time_of_first_tweet_sin","time_of_first_tweet_cos",
    "time_of_last_tweet_sin","time_of_last_tweet_cos",
]
STANDARDIZE = [ # puts values on a common scale by subtracting the mean and dividing by the standard deviation to remove units
    "avg_tweet_length","tweet_entropy",
]
LOGSTD_COUNTS = [ # applies a log transformation to count data to stabilize variance and then standardizes it
    "tweets_total","retweets_total","active_hours_count",
    "morning_tweets","afternoon_tweets","evening_tweets","night_tweets",
    "burst_count","max_burst_length","7d_avg_tweets","7d_var_tweets",
]
LOGSTD_GAPS = [ # log-transforms measures of gaps or distances to reduce skew, then standardizes them to the same statistical scale
    "mean_intertweet_gap","std_intertweet_gap","min_gap","max_gap",
]


log1p_std = Pipeline([
    ("log1p", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
    ("std", StandardScaler())
])
std = Pipeline([("std", StandardScaler())])


df = pd.read_csv(CSV_PATH, parse_dates=[0])
df = df.sort_values(df.columns[0]).reset_index(drop=True)
date_col = df.columns[0]

for c in df.columns[1:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# sklearn ColumnTransformer object does work for us
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


# select range of data points (rows) to normalize the data on
n = len(df)
split = max(1, int(n * TRAIN_FRAC))
X_train = df.iloc[:split, 1:]  
X_all   = df.iloc[:, 1:]

# fit the normalization
ct.fit(X_train)

X_scaled = ct.transform(X_all)
feat_names = ct.get_feature_names_out()
df_scaled = pd.DataFrame(X_scaled, columns=feat_names, index=df.index)

# Keep date column as is
out = pd.concat([df[[date_col]].reset_index(drop=True), df_scaled.reset_index(drop=True)], axis=1)
out.to_csv(OUT_CSV, index=False)
joblib.dump(ct, PREPROC_PATH)


