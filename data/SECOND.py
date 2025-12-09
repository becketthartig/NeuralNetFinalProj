import pandas as pd
import numpy as np

# load in file that FIRST.py produces
df = pd.read_csv("tweettimes.csv", names=["date","time","char_count","is_RT"], header=0)

# here is the logic to parse the timestamps in the original file into something
# pandas can work with to develop temporal features
df["time_clean"] = df["time"].astype(str).str.replace(r"\s+[A-Z]{2,4}$", "", regex=True)

ts_str = df["date"].astype(str).str.strip() + " " + df["time_clean"].astype(str).str.strip()
df["ts"] = pd.to_datetime(ts_str, format="%b %d %Y %I:%M:%S %p", errors="coerce")
if df["ts"].isna().any():
    bad = df.loc[df["ts"].isna(), ["date","time"]].head()
    raise ValueError(f"Unparsed timestamps, e.g.\n{bad}")
df["ts"] = df["ts"].dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="infer")

df["char_count"] = df["char_count"].astype(int)
df["is_RT"] = df["is_RT"].astype(int)
df = df.sort_values("ts").reset_index(drop=True)
df["hour"] = df["ts"].dt.hour
df["date_local"] = df["ts"].dt.date

# Helper functions to create individual features
# Can each be applied to a column in the dataframe

def entropy_from_hist(counts): # Shannon entropy on hour probabilities (see readme.md
    tot = counts.sum()
    if tot == 0: return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def circ_from_time(ts): # sine and cos features (kind of useless)
    frac = (ts.hour + ts.minute/60 + ts.second/3600) / 24.0
    return np.sin(2*np.pi*frac), np.cos(2*np.pi*frac)

def day_bin(h):
    if h < 6: return "night"
    if h < 12: return "morning"
    if h < 18: return "afternoon"
    return "evening"

df["day_bin"] = df["hour"].apply(day_bin)

# intertweet gap (minutes) within each day
df["gap_min"] = df.groupby("date_local")["ts"].diff().dt.total_seconds() / 60.0

BURST_GAP_MIN = 10 # 10 minutes as burst threshold

def per_day(g):
    total = len(g)
    rts   = int(g["is_RT"].sum())
    orig  = total - rts
    tweet_ratio = orig / max(orig + rts, 1)

    avg_tph = total / 24.0
    active_hours = int(g["hour"].nunique())

    # create intertweet gap features
    gaps = g["gap_min"].dropna().to_numpy()
    mean_gap = float(np.mean(gaps)) if gaps.size else np.nan
    std_gap  = float(np.std(gaps)) if gaps.size else np.nan
    min_gap  = float(np.min(gaps)) if gaps.size else np.nan
    max_gap  = float(np.max(gaps)) if gaps.size else np.nan

    # define a burst
    # see a tweet and see if another tweet comes withing 10 minutes of it
    if total > 0:
        over = (g["gap_min"].fillna(np.inf).to_numpy() > BURST_GAP_MIN).astype(int)
        burst_id = np.cumsum(np.insert(over, 0, 1))[:-1]
        sizes = pd.Series(burst_id).value_counts()
        burst_count = int(sizes.size)
        max_burst_len = int(sizes.max())
    else:
        burst_count = 0
        max_burst_len = 0

    # burstiness = (std−mean)/(std+mean) on intertweet caps gaps
    if gaps.size >= 2 and (np.mean(gaps) + np.std(gaps)) > 0:
        burstiness = float((np.std(gaps) - np.mean(gaps)) / (np.std(gaps) + np.mean(gaps)))
    else:
        burstiness = np.nan

    avg_len = float(g["char_count"].mean()) if total else 0.0

    vc = g["day_bin"].value_counts()
    morning = int(vc.get("morning", 0))
    afternoon = int(vc.get("afternoon", 0))
    evening = int(vc.get("evening", 0))
    night = int(vc.get("night", 0))

    # see which hours a tweet was posted in to calculate entropy
    hour_hist = g["hour"].value_counts().reindex(range(24), fill_value=0).sort_index().to_numpy()
    tweet_entropy = entropy_from_hist(hour_hist)

    first_ts = g["ts"].min()
    last_ts  = g["ts"].max()
    f_sin, f_cos = circ_from_time(first_ts)
    l_sin, l_cos = circ_from_time(last_ts)

    return pd.Series({
        "tweets_total": total,
        "retweets_total": rts,
        "tweet_ratio": tweet_ratio,
        "avg_tweets_per_hour": avg_tph,
        "active_hours_count": active_hours,
        "mean_intertweet_gap": mean_gap,
        "std_intertweet_gap": std_gap,
        "min_gap": min_gap,
        "max_gap": max_gap,
        "burst_count": burst_count,
        "max_burst_length": max_burst_len,
        "burstiness": burstiness,
        "avg_tweet_length": avg_len,
        "morning_tweets": morning,
        "afternoon_tweets": afternoon,
        "evening_tweets": evening,
        "night_tweets": night,
        "tweet_entropy": tweet_entropy,
        "time_of_first_tweet_sin": f_sin,
        "time_of_first_tweet_cos": f_cos,
        "time_of_last_tweet_sin":  l_sin,
        "time_of_last_tweet_cos":  l_cos,
    })

# Consolidate all tweet rows into one per day
daily = df.groupby("date_local", as_index=False).apply(per_day).reset_index(drop=True)

# apply windowed pandas mean and variance for 7 day rolling windows
daily = daily.sort_values("date_local")
daily["7d_avg_tweets"] = daily["tweets_total"].rolling(window=7, min_periods=1).mean()
daily["7d_var_tweets"] = daily["tweets_total"].rolling(window=7, min_periods=2).var(ddof=0)

daily.to_csv("daily_features.csv", index=False)
