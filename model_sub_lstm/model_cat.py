#!/usr/bin/env python3
"""
Participant code-submission baseline.
Usage (called by ingestion):
python -m model --train_hourly <csv> --test_hourly <csv> --test_index <csv> --predictions_out <csv>
"""
import argparse, pandas as pd, numpy as np, pickle, os
from pathlib import Path
from catboost import CatBoostClassifier
_HAS_CATBOOST = True
HIST_DAYS=7; FUTURE_DAYS=14
FEATURES = ["sea_level", "sea_level_3d_mean", "sea_level_7d_mean", "sea_level_std"]

def daily_aggregate(df):
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["date"] = df["time"].dt.floor("D")
    daily = (df.groupby(["station_name","date"])
               .agg(sea_level=("sea_level","mean"),
                    sea_level_max=("sea_level","max"),
                    latitude=("latitude","first"),
                    longitude=("longitude","first"))
               .reset_index())
    daily = daily.sort_values(["station_name","date"]).reset_index(drop=True)
    # Enhanced feature engineering - multiple rolling windows
    daily["sea_level_3d_mean"] = (daily.groupby("station_name")["sea_level"]
                                  .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    daily["sea_level_7d_mean"] = (daily.groupby("station_name")["sea_level"]
                                  .transform(lambda x: x.rolling(7, min_periods=1).mean()))
    daily["sea_level_std"] = (daily.groupby("station_name")["sea_level"]
                              .transform(lambda x: x.rolling(7, min_periods=1).std().fillna(0)))
    return daily

def build_windows(daily, stations, use_labels=False, thresholds=None):
    X, y, meta = [], [], []
    if use_labels:
        if thresholds is None:
            thr = (daily.groupby("station_name")["sea_level"]
                        .agg(["mean","std"]).assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
                        [["flood_threshold"]].reset_index())
        else:
            thr = thresholds
        daily = daily.merge(thr, on="station_name", how="left")
        daily["flood"] = (daily["sea_level_max"] > daily["flood_threshold"]).astype(int)

    for stn, grp in daily[daily["station_name"].isin(stations)].groupby("station_name"):
        grp = grp.sort_values("date").reset_index(drop=True)
        for i in range(len(grp) - HIST_DAYS - FUTURE_DAYS + 1):
            hist_block = grp.loc[i:i+HIST_DAYS-1, FEATURES]
            if hist_block.isna().any().any():
                continue
            # For CatBoost, flatten the sequence to a 1D feature vector
            X.append(hist_block.values.flatten())
            meta.append({"station": stn,
                         "hist_start": grp.loc[i, "date"],
                         "future_start": grp.loc[i+HIST_DAYS, "date"]})
            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"]
                # For binary classification, use max flood indicator across future days
                y.append(int(fut.max() > 0))
    return np.array(X), (np.array(y) if use_labels else None), pd.DataFrame(meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    train = pd.read_csv(args.train_hourly)
    test  = pd.read_csv(args.test_hourly)
    index = pd.read_csv(args.test_index)

    daily_te = daily_aggregate(test)

    # Check if model.pkl exists - if so, load pre-trained model; otherwise train new one
    model_pkl_path = Path("model.pkl")
    if model_pkl_path.exists():
        # Load pre-trained model
        print("Loading pre-trained model from model.pkl...")
        with open(model_pkl_path, 'rb') as f:
            model_dict = pickle.load(f)
        
        clf = model_dict['model']
        print("Pre-trained model loaded successfully.")
    else:
        # Train new model (backward compatibility)
        print("model.pkl not found. Training new model...")
        daily_tr = daily_aggregate(train)

        # thresholds from TRAIN only
        thr = (train.groupby("station_name")["sea_level"]
                    .agg(["mean","std"])
                    .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
                    [["flood_threshold"]]
                    .reset_index())

        stn_tr = daily_tr["station_name"].unique().tolist()
        X_tr, y_tr, _ = build_windows(daily_tr, stn_tr, use_labels=True, thresholds=thr)

        # Calculate scale_pos_weight for imbalanced classes
        pos = int((y_tr==1).sum())
        neg = int((y_tr==0).sum())
        spw = float(neg/max(pos, 1))

        # Initialize CatBoost model
        clf = CatBoostClassifier(
            iterations=400,
            depth=6,
            learning_rate=0.05,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=42,
            scale_pos_weight=spw,
            verbose=False,
            task_type='CPU',
            early_stopping_rounds=50
        )

        # Train with validation split
        val_size = int(len(X_tr) * 0.1)
        train_size = len(X_tr) - val_size
        train_indices = np.arange(train_size)
        val_indices = np.arange(train_size, len(X_tr))

        clf.fit(
            X_tr[train_indices], 
            y_tr[train_indices],
            eval_set=(X_tr[val_indices], y_tr[val_indices]),
            plot=False
        )

    # Build test windows and align to index by key
    X_te, _, meta_te = build_windows(daily_te, daily_te["station_name"].unique().tolist(), use_labels=False)
    
    if len(X_te)==0:
        raise RuntimeError("No test windows could be built from provided test_hourly/test_index. Check dates.")

    # Predict using CatBoost
    probs = clf.predict_proba(X_te)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_te)

    meta_te["key"] = meta_te["station"].astype(str) + "|" + meta_te["hist_start"].astype(str) + "|" + meta_te["future_start"].astype(str)

    index["hist_start"] = pd.to_datetime(index["hist_start"])
    index["future_start"] = pd.to_datetime(index["future_start"])
    index["key"] = index["station_name"].astype(str) + "|" + index["hist_start"].astype(str) + "|" + index["future_start"].astype(str)

    pred_df = pd.DataFrame({"key": meta_te["key"], "y_prob": probs})

    out = index.merge(pred_df, on="key", how="left")[["id","y_prob"]]
    out["y_prob"] = out["y_prob"].fillna(0.5)
    out.to_csv(args.predictions_out, index=False)
    print(f"Wrote {args.predictions_out}")

if __name__ == "__main__":
    main()
