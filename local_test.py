import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

# --- Load
df = pd.read_csv("daily_features_normalized.csv")
dfy = pd.read_csv("daily_features.csv")

# --- Select features / target
X_cols = [
    'tweets_total', 'retweets_total','tweet_ratio','avg_tweets_per_hour',
    'mean_intertweet_gap','std_intertweet_gap',
    'avg_tweet_length','tweet_entropy',
    'time_of_first_tweet_sin','time_of_first_tweet_cos',
    'time_of_last_tweet_sin','time_of_last_tweet_cos',
    '7d_avg_tweets','7d_var_tweets'
]
X_cols = [
    "tweet_ratio","burstiness","time_of_first_tweet_sin","time_of_first_tweet_cos","time_of_last_tweet_sin","time_of_last_tweet_cos","avg_tweet_length","tweet_entropy","tweets_total","retweets_total","active_hours_count","morning_tweets","afternoon_tweets","evening_tweets","night_tweets","burst_count","max_burst_length","7d_avg_tweets","7d_var_tweets","mean_intertweet_gap","std_intertweet_gap","min_gap","max_gap"
]

y_col = 'tweets_total'

# --- Clean: coerce numeric, drop rows with NaN target, fill feature NaNs
X = df[X_cols].apply(pd.to_numeric, errors='coerce')
y = pd.to_numeric(dfy[y_col], errors='coerce')
mask = y.notna()
X, y = X.loc[mask].copy(), y.loc[mask].copy()

# Fill remaining NaNs in features with column medians (simple + safe)
X = X.fillna(X.median())

# Optional: clip extreme outliers to reduce blow-ups
# X = X.clip(X.quantile(0.001), X.quantile(0.999), axis=1)

# --- Split (chronological split is better for time series; here: random for simplicity)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Scale features (fit on train only)
# scaler = StandardScaler()
# X_train_s = scaler.fit_transform(X_train)
# X_test_s  = scaler.transform(X_test)

# --- Model: small MLP for regression
tf.random.set_seed(42)
model = keras.Sequential([
    layers.Input(shape=(X_train.shape[1],)),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(32, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
    layers.Dense(1, activation='linear')
])

model.compile(optimizer='adam',
              loss='mse',
              metrics=[keras.metrics.MAE, keras.metrics.RootMeanSquaredError()])

cb = [keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
history = model.fit(X_train, y_train, validation_split=0.2, epochs=200, batch_size=32, callbacks=cb)

# --- Eval
test_loss, test_mae, test_rmse = model.evaluate(X_test, y_test, verbose=0)
print(f"Test MAE:  {test_mae:.2f}")
print(f"Test RMSE: {test_rmse:.2f}")


import matplotlib.pyplot as plt

date_col = df.columns[0]                          # first column is the date
dates = pd.to_datetime(df[date_col])            # use the TRAIN-FIT scaler
y_pred_all = model.predict(X, verbose=0).ravel()

# Build a results frame and mark train/test rows
res = pd.DataFrame({
    "date": dates,
    "y_true": y.values,
    "y_pred": y_pred_all,
})
res["split"] = "train"
res.loc[X_test.index, "split"] = "test"

# Sort by time
res = res.sort_values("date").reset_index(drop=True)

# --- Plot actual vs predicted over time ---
plt.figure(figsize=(12,4.5))
plt.plot(res["date"], dfy['7d_avg_tweets'], label="7 day avg", linewidth=1.5)
plt.plot(res["date"], res["y_pred"], label="Predicted", linewidth=1.5)
plt.plot(res["date"], y, label="Actual", linewidth=1.2)

# Lightly highlight test region(s)
is_test = res["split"].eq("test").to_numpy()
# find contiguous test segments
starts = np.where((~is_test[:-1] & is_test[1:]))[0] + 1
ends   = np.where((is_test[:-1] & ~is_test[1:]))[0] + 1
if is_test[0]:  starts = np.r_[0, starts]
if is_test[-1]: ends   = np.r_[ends, len(is_test)]
for s,e in zip(starts, ends):
    plt.axvspan(res["date"].iloc[s], res["date"].iloc[e-1], alpha=0.08)

plt.title("Tweet count: Actual vs Predicted")
plt.xlabel("Date"); plt.ylabel("Tweets per day")
plt.legend(); plt.grid(True, alpha=0.3); plt.gcf().autofmt_xdate()
plt.tight_layout()
plt.savefig("pred_vs_actual.png", dpi=150)
plt.show()

# Optional: quick error stats
res["abs_err"] = (res["y_pred"] - res["y_true"]).abs()
print("Overall MAE:", res["abs_err"].mean().round(2))
print("Test MAE   :", res.loc[res["split"]=="test", "abs_err"].mean().round(2))

from sklearn.decomposition import PCA

hidden_layer = model.layers[-2].name  # or use an index like model.layers[-2]
feat_model = keras.Model(inputs=model.inputs, outputs=model.get_layer(hidden_layer).output)

H_all = feat_model.predict(X)  # (N, hidden_dim)

pca = PCA(n_components=2, random_state=42)
Z = pca.fit_transform(H_all)  # (N, 2)

print("Activation PCA explained variance:", pca.explained_variance_ratio_)
plt.figure(figsize=(6,5))
for split, m in [("train","o"), ("test","s")]:
    idx = res["split"].values == split
    plt.scatter(Z[idx,0], Z[idx,1], s=18, alpha=0.7, label=split, marker=m)
plt.title("PCA of hidden activations")
plt.xlabel("PC1"); plt.ylabel("PC2"); plt.legend(); plt.tight_layout()
plt.savefig("pca_activations.png", dpi=150); plt.show()

from sklearn.metrics import mean_absolute_error

base_pred = model.predict(X_test, verbose=0).ravel()
base_mae = mean_absolute_error(y_test, base_pred)

def perm_importance(X, y, n_repeats=10, seed=42):
    rng = np.random.default_rng(seed)
    imp = []
    for j, col in enumerate(X.columns):
        deltas = []
        Xp = X.copy()
        for _ in range(n_repeats):
            Xp_col = Xp.iloc[:, j].to_numpy().copy()
            rng.shuffle(Xp_col)
            Xp.iloc[:, j] = Xp_col
            preds = model.predict(Xp, verbose=0).ravel()
            deltas.append(mean_absolute_error(y, preds) - base_mae)
            Xp.iloc[:, j] = X.iloc[:, j]  # restore
        imp.append(np.mean(deltas))
    return pd.Series(imp, index=X.columns).sort_values(ascending=False)

imp = perm_importance(X_test.copy(), y_test)
print(imp.head(10))