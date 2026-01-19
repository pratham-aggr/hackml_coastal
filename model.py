#!/usr/bin/env python3
"""
Lightweight submission model for coastal flood prediction.

Usage (called by the ingestion program):
  python model.py --train_hourly <train_hourly.csv> \
                  --test_hourly <test_hourly.csv> \
                  --test_index  <test_index.csv> \
                  --predictions_out <predictions.csv>

This script trains a RandomForestClassifier (no GridSearchCV) to predict
the probability that any flood occurs in the next 14 days given a 7-day
historical window of daily mean sea level.
"""
import argparse
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle


HIST_DAYS = 7
FUTURE_DAYS = 14


def read_hourly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "time" in df.columns:
        df["time"] = pd.to_datetime(df["time"]).dt.floor("D")
    else:
        # try common alternatives
        if "date" in df.columns:
            df["time"] = pd.to_datetime(df["date"]).dt.floor("D")
        else:
            raise ValueError("Hourly CSV must contain a 'time' or 'date' column")
    return df


def compute_thresholds_from_hourly(df_hourly: pd.DataFrame) -> pd.Series:
    # threshold = mean + k * std (computed on hourly sea level per station)
    # use a larger default multiplier to reduce excessive flood labels in short demos
    g = df_hourly.groupby("station_name")["sea_level"]
    stats = g.agg(["mean", "std"]).fillna(0.0)
    thresh = stats["mean"] + 3.0 * stats["std"]
    return thresh.to_dict()


def hourly_to_daily(df_hourly: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    # aggregate to daily mean, daily max, min, and std
    df = df_hourly.copy()
    df = df.groupby(["station_name", "time"]).agg(
        sea_level=("sea_level", "mean"),
        sea_level_max=("sea_level", "max"),
        sea_level_min=("sea_level", "min"),
        sea_level_std=("sea_level", "std")
    ).reset_index()
    # add flood flag using provided thresholds
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
                # basic aggregated stats
                f_mean = float(np.nanmean(mean_arr))
                f_max = float(np.nanmax(max_arr))
                f_min = float(np.nanmin(min_arr))
                f_std = float(np.nanstd(mean_arr))
                # slope (trend) across the HIST_DAYS window
                try:
                    xs = np.arange(len(mean_arr))
                    if np.all(np.isfinite(mean_arr)):
                        slope = float(np.polyfit(xs, mean_arr, 1)[0])
                    else:
                        slope = 0.0
                except Exception:
                    slope = 0.0
                f_last = float(mean_arr[-1])
                f_diff_mean = float(np.mean(np.diff(mean_arr))) if len(mean_arr) > 1 else 0.0
                f_max_over_mean = float(f_max / (f_mean + 1e-6))
                # flood-related counts and recency
                flood_arr = hist.get("flood", pd.Series([0]*len(hist))).values
                f_count_floods = int(np.sum(flood_arr))
                # days since last flood within the window (0 means last day flooded)
                try:
                    last_flood_idxs = np.where(flood_arr == 1)[0]
                    if len(last_flood_idxs) == 0:
                        days_since_last = HIST_DAYS
                    else:
                        days_since_last = int((HIST_DAYS - 1) - last_flood_idxs[-1])
                except Exception:
                    days_since_last = HIST_DAYS
                # cyclical day-of-year features using the last day in the window
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


def build_test_features(test_daily: pd.DataFrame, test_index: pd.DataFrame):
    rows = []
    for _, r in test_index.iterrows():
        sid = r.get("id")
        stn = r.get("station_name")
        hist_start = pd.to_datetime(r.get("hist_start"))
        hist_end = pd.to_datetime(r.get("hist_end"))
        # select rows for station between hist_start and hist_end inclusive
        g = test_daily[(test_daily["station_name"] == stn) &
                       (test_daily["time"] >= hist_start) &
                       (test_daily["time"] <= hist_end)].sort_values("time")
        # build or pad arrays to HIST_DAYS
        if len(g) == 0:
            # create a placeholder row with zeros and a fake time
            times = [hist_end] * HIST_DAYS
            mean_arr = np.zeros(HIST_DAYS)
            max_arr = np.zeros(HIST_DAYS)
            min_arr = np.zeros(HIST_DAYS)
            flood_arr = np.zeros(HIST_DAYS, dtype=int)
        else:
            # extract arrays and pad with last available values if needed
            mean_vals = g["sea_level"].values
            max_vals = g.get("sea_level_max", g["sea_level"]).values
            min_vals = g.get("sea_level_min", g["sea_level"]).values
            flood_vals = g.get("flood", pd.Series([0]*len(g))).values
            if len(mean_vals) < HIST_DAYS:
                pad_len = HIST_DAYS - len(mean_vals)
                mean_arr = np.concatenate([mean_vals, np.full(pad_len, mean_vals[-1])])
                max_arr = np.concatenate([max_vals, np.full(pad_len, max_vals[-1])])
                min_arr = np.concatenate([min_vals, np.full(pad_len, min_vals[-1])])
                flood_arr = np.concatenate([flood_vals, np.zeros(pad_len, dtype=int)])
                times = list(pd.to_datetime(g["time"]).tolist()) + [pd.to_datetime(g["time"]).tolist()[-1]] * pad_len
            else:
                mean_arr = mean_vals[-HIST_DAYS:]
                max_arr = max_vals[-HIST_DAYS:]
                min_arr = min_vals[-HIST_DAYS:]
                flood_arr = flood_vals[-HIST_DAYS:]
                times = pd.to_datetime(g["time"]).tolist()[-HIST_DAYS:]

        # compute features same as training
        f_mean = float(np.nanmean(mean_arr))
        f_max = float(np.nanmax(max_arr))
        f_min = float(np.nanmin(min_arr))
        f_std = float(np.nanstd(mean_arr))
        try:
            xs = np.arange(len(mean_arr))
            slope = float(np.polyfit(xs, mean_arr, 1)[0]) if np.all(np.isfinite(mean_arr)) else 0.0
        except Exception:
            slope = 0.0
        f_last = float(mean_arr[-1])
        f_diff_mean = float(np.mean(np.diff(mean_arr))) if len(mean_arr) > 1 else 0.0
        f_max_over_mean = float(f_max / (f_mean + 1e-6))
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
            last_time = pd.to_datetime(times[-1])
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
        rows.append({"id": sid, "features": feat})

    ids = [r["id"] for r in rows]
    X = np.vstack([r["features"] for r in rows]) if rows else np.zeros((0, 12))
    return ids, X


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    ap.add_argument("--demo", action="store_true", help="Run a synthetic demo: train, evaluate, and save a model.pkl")
    args = ap.parse_args()

    train_hourly = Path(args.train_hourly)
    test_hourly = Path(args.test_hourly)
    test_index = Path(args.test_index)
    out_path = Path(args.predictions_out)

    train_h = read_hourly(train_hourly)
    test_h = read_hourly(test_hourly)
    idx = pd.read_csv(test_index)
    # ensure id column
    if "id" not in idx.columns:
        idx.insert(0, "id", range(len(idx)))

    thresholds = compute_thresholds_from_hourly(train_h)
    train_daily = hourly_to_daily(train_h, thresholds)
    test_daily = hourly_to_daily(test_h, thresholds)

    X_train, y_train = build_train_set(train_daily)
    if X_train.shape[0] == 0:
        raise RuntimeError("No training samples could be built from train_hourly input.")

    # Balance performance and runtime for submission environments:
    # - cap total training samples by stratified subsampling (preserve class balance)
    # - use a moderate RandomForest size but single-threaded to avoid worker overhead
    MAX_TRAIN_SAMPLES = 8000
    if X_train.shape[0] > MAX_TRAIN_SAMPLES:
        rng = np.random.RandomState(42)
        classes = np.unique(y_train)
        sel_idx = []
        per_class = max(1, MAX_TRAIN_SAMPLES // len(classes))
        for c in classes:
            idxs = np.where(y_train == c)[0]
            if len(idxs) > per_class:
                pick = rng.choice(idxs, per_class, replace=False)
            else:
                pick = idxs
            sel_idx.extend(pick.tolist())
        # if still under budget, fill with random remaining
        sel_idx = np.array(sel_idx, dtype=int)
        if len(sel_idx) < MAX_TRAIN_SAMPLES:
            remaining = MAX_TRAIN_SAMPLES - len(sel_idx)
            all_remaining = np.setdiff1d(np.arange(X_train.shape[0]), sel_idx)
            if len(all_remaining) > 0:
                add = rng.choice(all_remaining, min(remaining, len(all_remaining)), replace=False)
                sel_idx = np.concatenate([sel_idx, add])
        X_train = X_train[sel_idx]
        y_train = y_train[sel_idx]

    clf = RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, n_jobs=1, class_weight="balanced")
    clf.fit(X_train, y_train)

    ids, X_test = build_test_features(test_daily, idx)
    if X_test.shape[0] == 0:
        # create empty predictions with zeros
        preds_df = pd.DataFrame({"id": idx["id"].tolist(), "y_prob": [0.0] * len(idx)})
    else:
        try:
            probs = clf.predict_proba(X_test)[:, 1]
        except Exception:
            probs = clf.predict(X_test).astype(float)
        preds_df = pd.DataFrame({"id": ids, "y_prob": probs})

    out_path.parent.mkdir(parents=True, exist_ok=True)
    preds_df.to_csv(out_path, index=False)

    # save a compact model artifact next to predictions (optional)
    try:
        with open(out_path.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(clf, f)
    except Exception:
        pass


def demo_train_and_save(save_path: str = "model.pkl"):
    """Create a small synthetic dataset, train the model, evaluate, and save the trained model."""
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
    import datetime as dt
    import tempfile
    import os

    # generate synthetic hourly train/test CSVs similar to run_modelpy_rf
    rng = np.random.RandomState(1)
    n_stations = 6
    days_train = 90
    days_test = 30
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
                rows.append({"station_name": stn, "time": t, "sea_level": val, "latitude": 0.0, "longitude": 0.0})
    train_df = pd.DataFrame(rows)

    rows = []
    start_test = start + dt.timedelta(days=days_train)
    for stn in stations:
        for d in range(days_test):
            for h in range(24):
                t = start_test + dt.timedelta(days=d, hours=h)
                base = rng.normal(0, 1)
                val = base + 0.05 * rng.randn()
                if rng.rand() < 0.02:
                    val += 2.5 + 0.5 * rng.randn()
                rows.append({"station_name": stn, "time": t, "sea_level": val, "latitude": 0.0, "longitude": 0.0})
    test_df = pd.DataFrame(rows)

    td = tempfile.mkdtemp()
    train_csv = os.path.join(td, "train_hourly.csv")
    test_csv = os.path.join(td, "test_hourly.csv")
    train_df.to_csv(train_csv, index=False)
    test_df.to_csv(test_csv, index=False)

    # use pipeline helpers to construct windows
    train_h = read_hourly(train_csv)
    test_h = read_hourly(test_csv)
    thresholds = compute_thresholds_from_hourly(train_h)
    train_daily = hourly_to_daily(train_h, thresholds)
    test_daily = hourly_to_daily(test_h, thresholds)

    X, y = build_train_set(train_daily)
    if X.shape[0] == 0:
        raise RuntimeError("No training samples could be built from synthetic demo data.")

    # split into train/validation/test (60/20/20)
    X_tr, X_tmp, y_tr, y_tmp = train_test_split(X, y, test_size=0.4, random_state=42, stratify=y)
    X_val, X_te, y_val, y_te = train_test_split(X_tmp, y_tmp, test_size=0.5, random_state=42, stratify=y_tmp)

    # train a RandomForest similar to run_modelpy_rf but single-threaded for submission safety
    clf = RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42, n_jobs=1, class_weight="balanced")
    clf.fit(X_tr, y_tr)

    yval_pred = clf.predict(X_val)
    yte_pred = clf.predict(X_te)

    val_acc = accuracy_score(y_val, yval_pred)
    val_f1 = f1_score(y_val, yval_pred)
    try:
        val_mcc = matthews_corrcoef(y_val, yval_pred)
    except Exception:
        val_mcc = float('nan')

    te_acc = accuracy_score(y_te, yte_pred)
    te_f1 = f1_score(y_te, yte_pred)
    try:
        te_mcc = matthews_corrcoef(y_te, yte_pred)
    except Exception:
        te_mcc = float('nan')

    print(f"Demo training complete.")
    print(f"Validation — Accuracy: {val_acc:.4f}, F1: {val_f1:.4f}, MCC: {val_mcc:.4f}")
    print(f"Test —       Accuracy: {te_acc:.4f}, F1: {te_f1:.4f}, MCC: {te_mcc:.4f}")

    with open(save_path, 'wb') as f:
        pickle.dump(clf, f)
    print(f"Saved demo model to {save_path}")


if __name__ == "__main__":
    # allow a quick demo mode for reproducible training/evaluation
    import sys
    args = sys.argv[1:]
    if "--demo" in args:
        demo_train_and_save("model.pkl")
    else:
        main()