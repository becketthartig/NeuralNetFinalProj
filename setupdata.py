import csv
from collections import defaultdict
import numpy as np
import tensorflow as tf

days = []

def csv_to_grouped_dict(path, include_cols):
    """
    Reads a CSV, groups rows by the first column, and keeps only the
    columns specified in include_cols.

    Args:
        path (str): path to CSV file
        include_cols (list[int]): indices of columns to keep 
                                  (0-based indexing)

    Returns:
        dict[str, list[list[str]]]
    """
    groups = defaultdict(list)

    with open(path, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)  # always skip header

        # print([header[i] for i in include_cols])

        for row in reader:
            if not row:
                continue

            key = row[0]
            # extract only requested columns
            selected = [float(row[i]) if row[i] != "" else 0.0 for i in include_cols]
            if path == "daily_features.csv":
                days.append(row[0])
                selected = [0.0, 0.0] + selected

            groups[key].append(selected)
    return dict(groups)


daily_features = csv_to_grouped_dict("daily_features.csv",
                                     [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 18, 23, 24])
tweets_with_partial_features = csv_to_grouped_dict("tweets_with_partial_features.csv",
                                                   [4, 5, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22])
tweettimes6 = csv_to_grouped_dict("tweettimes6.csv",
                                  [12])

# print(daily_features)
daysED = days[7:-8]





def build_lstm_dataset(all_days, daily_features, partial_features, targets,
                       num_partial_samples=3, max_windows=None, seed=42):

    rng = np.random.default_rng(seed)

    X_seqs = []
    y_vals = []

    # we need windows of 7 consecutive days -> indices 0..len-1
    # window: days[i-6 : i+1] with i starting from 6
    for i in range(6, len(all_days)):
        if max_windows is not None and len(X_seqs) >= max_windows:
            break

        window_days = all_days[i-6:i+1]  # 6 full days + 1 partial day
        full_days = window_days[:6]
        partial_day = window_days[6]

    #     # check that all days are present in daily_features
        if any(d not in daily_features for d in full_days):
            continue

    #     # partial day must exist and have partial snapshots and targets
        if partial_day not in partial_features:
            continue
        if partial_day not in targets:
            continue

        partial_list = partial_features[partial_day]
        target_list = targets[partial_day]

        if len(partial_list) == 0:
            continue

    #     # choose up to num_partial_samples indices without replacement
        n_partials = len(partial_list)
        n_sample = min(num_partial_samples, n_partials)
        sampled_indices = rng.choice(n_partials, size=n_sample, replace=False)

    #     # pre-build the 6 full-day part of the sequence
        full_stack = [np.asarray(daily_features[d], dtype=float) for d in full_days]


        for idx in sampled_indices:
            part_vec = np.asarray([partial_list[idx]], dtype=float)
            y = float(target_list[idx][0])

            # concatenate 6 full days + 1 partial snapshot
            seq = full_stack + [part_vec]
            seq = np.stack(seq, axis=0)  # shape (7, feature_dim)

            X_seqs.append(seq)
            y_vals.append(y)

    if not X_seqs:
        raise ValueError("No training samples could be constructed. Check your data/dicts.")

    X = np.stack(X_seqs, axis=0)  # (num_samples, 7, feature_dim)
    y = np.array(y_vals, dtype=float)  # (num_samples,)
    return np.squeeze(X, axis=2), y

X, y = build_lstm_dataset(
    all_days=daysED[0:7],
    daily_features=daily_features,
    partial_features=tweets_with_partial_features,
    targets=tweettimes6,
    num_partial_samples=3,
    max_windows=None,
    seed=123
)

print(X[0])
print(y)
print(X.shape)