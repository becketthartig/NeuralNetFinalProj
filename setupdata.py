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
                                  [11])

# print(daily_features)
daysED = days[8:-9]





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
    all_days=daysED,
    daily_features=daily_features,
    partial_features=tweets_with_partial_features,
    targets=tweettimes6,
    num_partial_samples=4,
    max_windows=None,
    seed=16
)

y = np.where(y >= 500, 25, y / 20)
y = np.trunc(y).astype(int)

print(X[0])
print(y)
print(X.shape)
# np.random.shuffle(X)


# lstm_model = tf.keras.models.Sequential([
#     # Shape [batch, time, features] => [batch, time, lstm_units]
#     tf.keras.layers.LSTM(32, return_sequences=True),
#     # Shape => [batch, time, features]
#     tf.keras.layers.Dense(units=1)
# ])

lstm_model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=(7, 18)),
    tf.keras.layers.LSTM(32, return_sequences=True),
    tf.keras.layers.LSTM(32, return_sequences=False),
    tf.keras.layers.Dense(26, activation='softmax')
])

lstm_model.compile(optimizer="adam", 
                   loss="sparse_categorical_crossentropy", 
                   metrics=["sparse_categorical_accuracy"])

def time_series_train_val_test_split(X, y, train_frac=0.6, val_frac=0.2):
    """
    X: (N, T, F)
    y: (N,)
    Splits WITHOUT shuffling (important for time series).
    """

    N = len(X)
    train_end = int(N * train_frac)
    val_end   = int(N * (train_frac + val_frac))

    X_train = X[:train_end]
    y_train = y[:train_end]

    X_val   = X[train_end:val_end]
    y_val   = y[train_end:val_end]

    X_test  = X[val_end:]
    y_test  = y[val_end:]

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

(X_train, y_train), (X_val, y_val), (X_test, y_test) = \
    time_series_train_val_test_split(X, y)


early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",        # what to watch
    patience=5,                # epochs with no improvement before stopping
    restore_best_weights=True # go back to best model (not last epoch)
)

lstm_model.fit(
    X_train, y_train,
    epochs=400,
    batch_size=32,
    verbose=1,
    validation_data=(X_val, y_val),
    callbacks=[early_stop]
)

# Evaluate on held-out test set
test_loss, test_mae = lstm_model.evaluate(X_test, y_test)
print("Test MSE:", test_loss)
print("Test MAE:", test_mae)


X_all = np.concatenate([X_train, X_val, X_test], axis=0)
y_all = np.concatenate([y_train, y_val, y_test], axis=0)

# y_all_pred = lstm_model.predict(X_all).flatten()
y_all_pred = lstm_model.predict(X_all).argmax(axis=1)

print(y_all.shape)
print(y_all_pred.shape)

import matplotlib.pyplot as plt

# --- Parity plot: y_true vs y_pred ---
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.scatter(y_all, y_all_pred, alpha=0.6)
min_val = min(y_all.min(), y_all_pred.min())
max_val = max(y_all.max(), y_all_pred.max())
plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("True target")
plt.ylabel("Predicted target")
plt.title("Predicted vs True (Test Set)")

# --- Time-series-style plot: index vs values ---
plt.subplot(1, 2, 2)
plt.plot(y_all, label="True", linewidth=1)
plt.plot(y_all_pred, label="Predicted", linewidth=1)
plt.xlabel("Sample index (test set)")
plt.ylabel("Target")
plt.title("Predicted and True over Test Samples")
plt.legend()

plt.tight_layout()
plt.show()