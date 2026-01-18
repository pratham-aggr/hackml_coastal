#!/usr/bin/env python3
"""
Compute train and test metrics for the model
"""
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, matthews_corrcoef
from pathlib import Path

# Read the processed data
train_hourly = pd.read_csv("processed_data/train_hourly.csv")
test_hourly = pd.read_csv("processed_data/test_hourly.csv")
test_index = pd.read_csv("processed_data/test_index.csv")
predictions = pd.read_csv("processed_data/predictions.csv")

print("=" * 60)
print("COMPUTING TRAINING METRICS")
print("=" * 60)

# Replicate the model's data processing
HIST_DAYS = 7
FUTURE_DAYS = 14
FEATURES = ["sea_level", "sea_level_3d_mean", "sea_level_7d_mean"]

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
    daily["sea_level_3d_mean"] = (daily.groupby("station_name")["sea_level"]
                                  .transform(lambda x: x.rolling(3, min_periods=1).mean()))
    daily["sea_level_7d_mean"] = (daily.groupby("station_name")["sea_level"]
                                  .transform(lambda x: x.rolling(7, min_periods=1).mean()))
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
            X.append(hist_block.values.flatten())
            meta.append({"station": stn,
                         "hist_start": grp.loc[i, "date"],
                         "future_start": grp.loc[i+HIST_DAYS, "date"]})
            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"]
                y.append(int(fut.max()>0))
    return np.array(X), (np.array(y) if use_labels else None), pd.DataFrame(meta)

# Process training data
daily_tr = daily_aggregate(train_hourly)
thr = (train_hourly.groupby("station_name")["sea_level"]
            .agg(["mean","std"])
            .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
            [["flood_threshold"]]
            .reset_index())

stn_tr = daily_tr["station_name"].unique().tolist()
X_tr, y_tr, _ = build_windows(daily_tr, stn_tr, use_labels=True, thresholds=thr)

print(f"Training samples: {len(X_tr)}")
print(f"Training positive samples: {y_tr.sum()}")
print(f"Training negative samples: {(y_tr==0).sum()}")
print(f"Training class balance: {y_tr.mean():.4f}")

# Train model to get predictions
try:
    from xgboost import XGBClassifier
    _HAS_XGB = True
except Exception:
    from sklearn.ensemble import GradientBoostingClassifier as XGBClassifier
    _HAS_XGB = False

pos = int((y_tr==1).sum())
neg = int((y_tr==0).sum())
spw = float(neg/max(pos,1))
clf = XGBClassifier(random_state=42, **({"n_estimators":400} if _HAS_XGB else {}))
if _HAS_XGB:
    clf.set_params(max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                   reg_lambda=1.0, reg_alpha=0.0, objective="binary:logistic", eval_metric="auc",
                   n_jobs=-1, scale_pos_weight=spw)
clf.fit(X_tr, y_tr)

# Training predictions
y_tr_pred_proba = clf.predict_proba(X_tr)[:,1] if hasattr(clf,"predict_proba") else clf.predict(X_tr)
y_tr_pred = (y_tr_pred_proba >= 0.5).astype(int)

train_auc = roc_auc_score(y_tr, y_tr_pred_proba)
train_acc = accuracy_score(y_tr, y_tr_pred)
train_f1 = f1_score(y_tr, y_tr_pred, zero_division=0)
train_mcc = matthews_corrcoef(y_tr, y_tr_pred)

print(f"\nTraining Metrics:")
print(f"  AUC:  {train_auc:.6f}")
print(f"  Accuracy: {train_acc:.6f}")
print(f"  F1 Score: {train_f1:.6f}")
print(f"  MCC:  {train_mcc:.6f}")

print("\n" + "=" * 60)
print("COMPUTING TEST METRICS")
print("=" * 60)

# Try to read reference data
try:
    y_test = pd.read_csv("Ingestion_Program/Reference data/y_test.csv", encoding='latin-1')
    print(f"Reference data shape: {y_test.shape}")
    print(f"Predictions shape: {predictions.shape}")
    
    # Merge predictions with ground truth (ensure id types match)
    y_test["id"] = y_test["id"].astype(int)
    predictions["id"] = predictions["id"].astype(int)
    merged = y_test.merge(predictions, on="id", how="inner")
    print(f"Merged data shape: {merged.shape}")
    
    if len(merged) > 0:
        y_true = merged["y_true"].astype(int).to_numpy()
        y_prob = merged["y_prob"].astype(float).to_numpy()
        y_pred = (y_prob >= 0.5).astype(int)
        
        test_auc = roc_auc_score(y_true, y_prob)
        test_acc = accuracy_score(y_true, y_pred)
        test_f1 = f1_score(y_true, y_pred, zero_division=0)
        test_mcc = matthews_corrcoef(y_true, y_pred)
        
        print(f"\nTest Metrics (on {len(merged)} samples):")
        print(f"  AUC:  {test_auc:.6f}")
        print(f"  Accuracy: {test_acc:.6f}")
        print(f"  F1 Score: {test_f1:.6f}")
        print(f"  MCC:  {test_mcc:.6f}")
    else:
        print("No overlapping IDs between reference data and predictions")
        print("Note: Reference data may be a sample subset")
except Exception as e:
    print(f"Could not compute test metrics: {e}")
    print("Note: Full test labels may not be available (competition setup)")

print("\n" + "=" * 60)
