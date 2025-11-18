import pandas as pd
import numpy as np

# ========== Load & parse ==========
# CSV columns: date,time,char_count,is_RT
# Example:
# Apr 18 2024,6:41:57 PM EDT,23,0
df = pd.read_csv("tweettimes.csv", names=["date","time","char_count","is_RT"], header=None)

# Strip trailing TZ (EST/EDT) from time for reliable parsing
df["time_clean"] = df["time"].astype(str).str.replace(r"\s+[A-Z]{2,4}$", "", regex=True)

# Join and parse -> timezone-naive
ts_str = df["date"].astype(str).str.strip() + " " + df["time_clean"].astype(str).str.strip()
df["ts"] = pd.to_datetime(ts_str, format="%b %d %Y %I:%M:%S %p", errors="coerce")
if df["ts"].isna().any():
    bad = df.loc[df["ts"].isna(), ["date","time"]].head()
    raise ValueError(f"Unparsed timestamps, e.g.\n{bad}")

# Localize to America/New_York (handles EST/EDT correctly)
df["ts"] = df["ts"].dt.tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="infer")

# Basic fields
df["char_count"] = df["char_count"].astype(int)
df["is_RT"] = df["is_RT"].astype(int)
df = df.sort_values("ts").reset_index(drop=True)
df["hour"] = df["ts"].dt.hour
df["date_local"] = df["ts"].dt.date

# ========== Helpers ==========
def entropy_from_hist(counts):
    tot = counts.sum()
    if tot == 0: return 0.0
    p = counts / tot
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())

def circ_from_time(ts):
    frac = (ts.hour + ts.minute/60 + ts.second/3600) / 24.0
    return np.sin(2*np.pi*frac), np.cos(2*np.pi*frac)

def day_bin(h):
    if h < 6: return "night"
    if h < 12: return "morning"
    if h < 18: return "afternoon"
    return "evening"

df["day_bin"] = df["hour"].apply(day_bin)

# Intertweet gap (minutes) within each day
df["gap_min"] = df.groupby("date_local")["ts"].diff().dt.total_seconds() / 60.0

BURST_GAP_MIN = 10  # minutes

def per_day(g):
    total = len(g)
    rts   = int(g["is_RT"].sum())
    orig  = total - rts
    tweet_ratio = orig / max(orig + rts, 1)

    avg_tph = total / 24.0
    active_hours = int(g["hour"].nunique())

    gaps = g["gap_min"].dropna().to_numpy()
    mean_gap = float(np.mean(gaps)) if gaps.size else np.nan
    std_gap  = float(np.std(gaps)) if gaps.size else np.nan
    min_gap  = float(np.min(gaps)) if gaps.size else np.nan
    max_gap  = float(np.max(gaps)) if gaps.size else np.nan

    # bursts: new burst when previous gap > threshold (first tweet starts a burst)
    if total > 0:
        over = (g["gap_min"].fillna(np.inf).to_numpy() > BURST_GAP_MIN).astype(int)
        burst_id = np.cumsum(np.insert(over, 0, 1))[:-1]
        sizes = pd.Series(burst_id).value_counts()
        burst_count = int(sizes.size)
        max_burst_len = int(sizes.max())
    else:
        burst_count = 0
        max_burst_len = 0

    # burstiness B = (σ−μ)/(σ+μ) on gaps
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

# ========== Aggregate ==========
daily = df.groupby("date_local", as_index=False).apply(per_day).reset_index(drop=True)

# 7d rolling stats (chronological)
daily = daily.sort_values("date_local")
daily["7d_avg_tweets"] = daily["tweets_total"].rolling(window=7, min_periods=1).mean()
daily["7d_var_tweets"] = daily["tweets_total"].rolling(window=7, min_periods=2).var(ddof=0)

daily.to_csv("daily_features.csv", index=False)
print("Wrote daily_features.csv with", len(daily), "rows.")