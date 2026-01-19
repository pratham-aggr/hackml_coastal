#!/usr/bin/env python3
"""
Train a lightweight logistic model using numpy/pandas (no scikit-learn required)
and save a pickled model to `model.pkl`.
"""
import numpy as np
import pandas as pd
import pickle
import datetime as dt
import os

HIST_DAYS = 7
FUTURE_DAYS = 14


def compute_thresholds_from_hourly(df_hourly: pd.DataFrame) -> dict:
    g = df_hourly.groupby("station_name")["sea_level"]
    stats = g.agg(["mean", "std"]).fillna(0.0)
    thresh = stats["mean"] + 3.0 * stats["std"]
    return thresh.to_dict()


def hourly_to_daily(df_hourly: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    df = df_hourly.copy()
    df = df.groupby(["station_name", "time"]).agg(
        sea_level=("sea_level", "mean"),
        sea_level_max=("sea_level", "max"),
        sea_level_min=("sea_level", "min"),
        sea_level_std=("sea_level", "std")
    ).reset_index()
    df["flood"] = df.apply(lambda r: 1 if r["sea_level_max"] > thresholds.get(r["station_name"], np.inf) else 0, axis=1)
    return df


def build_train_set(train_daily: pd.DataFrame):
    X = []
    y = []
    for stn, grp in train_daily.groupby("station_name"):
        g = grp.sort_values("time").reset_index(drop=True)
        for i in range(len(g) - HIST_DAYS - FUTURE_DAYS + 1):
            hist = g.loc[i:i+HIST_DAYS-1]
            future_block = g.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"].values
            if len(hist) == HIST_DAYS and len(future_block) == FUTURE_DAYS:
                mean_arr = hist["sea_level"].values
                max_arr = hist["sea_level_max"].values if "sea_level_max" in hist.columns else mean_arr
                min_arr = hist["sea_level_min"].values if "sea_level_min" in hist.columns else mean_arr
                f_mean = float(np.nanmean(mean_arr))
                f_max = float(np.nanmax(max_arr))
                f_min = float(np.nanmin(min_arr))
                f_std = float(np.nanstd(mean_arr))
                # slope
                try:
                    xs = np.arange(len(mean_arr))
                    slope = float(np.polyfit(xs, mean_arr, 1)[0]) if np.all(np.isfinite(mean_arr)) else 0.0
                except Exception:
                    slope = 0.0
                f_last = float(mean_arr[-1])
                f_diff_mean = float(np.mean(np.diff(mean_arr))) if len(mean_arr) > 1 else 0.0
                f_max_over_mean = float(f_max / (f_mean + 1e-6))
                flood_arr = hist.get("flood", pd.Series([0]*len(hist))).values
                f_count_floods = int(np.sum(flood_arr))
                try:
                    last_flood_idxs = np.where(flood_arr == 1)[0]
                    if len(last_flood_idxs) == 0:
                        days_since_last = HIST_DAYS
                    else:
                        days_since_last = int((HIST_DAYS - 1) - last_flood_idxs[-1])
                except Exception:
                    days_since_last = HIST_DAYS
                try:
                    last_time = pd.to_datetime(hist["time"].iloc[-1])
                    doy = float(last_time.timetuple().tm_yday)
                    doy_sin = float(np.sin(2 * np.pi * doy / 365.25))
                    doy_cos = float(np.cos(2 * np.pi * doy / 365.25))
                except Exception:
                    doy_sin = 0.0
                    doy_cos = 0.0
                feat = [
                    f_mean, f_max, f_min, f_std, slope, f_last,
                    f_diff_mean, f_max_over_mean, f_count_floods, days_since_last,
                    doy_sin, doy_cos
                ]
                X.append(feat)
                y.append(1 if future_block.max() > 0 else 0)
    if len(X) == 0:
        return np.zeros((0, 12)), np.zeros((0,))
    return np.vstack(X), np.array(y)


class SimpleLogisticModel:
    def __init__(self, coef, intercept, mean, std):
        self.coef_ = coef
        self.intercept_ = intercept
        self.mean_ = mean
        self.std_ = std

    def _scale(self, X):
        return (X - self.mean_) / (self.std_ + 1e-9)

    def predict_proba(self, X):
        Xs = self._scale(X)
        logits = Xs.dot(self.coef_) + self.intercept_
        probs = 1.0 / (1.0 + np.exp(-logits))
        return np.vstack([1-probs, probs]).T

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


def train_and_save(save_path="model.pkl", epochs=800, lr=0.1):
    # build synthetic dataset (reuse similar pattern as original demo)
    rng = np.random.RandomState(1)
    n_stations = 6
    days_train = 90
    stations = [f"S{i}" for i in range(n_stations)]
    start = dt.datetime(2000, 1, 1)
    rows = []
    for stn in stations:
        for d in range(days_train):
            for h in range(24):
                t = start + dt.timedelta(days=d, hours=h)
                base = rng.normal(0, 1)
                val = base + 0.05 * rng.randn()
                if rng.rand() < 0.02:
                    val += 2.5 + 0.5 * rng.randn()
                rows.append({"station_name": stn, "time": t, "sea_level": val})
    train_df = pd.DataFrame(rows)
    train_df["time"] = pd.to_datetime(train_df["time"]).dt.floor("D")

    thresholds = compute_thresholds_from_hourly(train_df)
    train_daily = hourly_to_daily(train_df, thresholds)
    X, y = build_train_set(train_daily)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples could be built from synthetic demo data.")

    # simple standardization
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    Xs = (X - mean) / (std + 1e-9)

    # initialize weights
    n_features = Xs.shape[1]
    w = np.zeros(n_features)
    b = 0.0

    # simple gradient descent logistic
    for epoch in range(epochs):
        logits = Xs.dot(w) + b
        probs = 1.0 / (1.0 + np.exp(-logits))
        # gradient
        error = probs - y
        grad_w = Xs.T.dot(error) / len(y)
        grad_b = error.mean()
        w -= lr * grad_w
        b -= lr * grad_b
        if epoch % 200 == 0:
            # simple loss
            loss = -np.mean(y * np.log(probs + 1e-12) + (1 - y) * np.log(1 - probs + 1e-12))
            print(f"epoch {epoch} loss {loss:.4f}")

    model = SimpleLogisticModel(coef=w, intercept=b, mean=mean, std=std)
    with open(save_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Saved model to {save_path}")


if __name__ == "__main__":
    train_and_save()
