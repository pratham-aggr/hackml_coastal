#!/usr/bin/env python3
"""Script to train CatBoost model and save to model.pkl"""
import pickle
import pandas as pd
import numpy as np
from model import daily_aggregate, build_windows, FEATURES
from catboost import CatBoostClassifier

# Load training data
train = pd.read_csv('../processed_data/train_hourly.csv')
print(f'Loaded {len(train)} rows')

# Preprocess
daily_tr = daily_aggregate(train)
thr = (train.groupby("station_name")["sea_level"]
            .agg(["mean","std"])
            .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
            [["flood_threshold"]]
            .reset_index())

stn_tr = daily_tr["station_name"].unique().tolist()
X_tr, y_tr, _ = build_windows(daily_tr, stn_tr, use_labels=True, thresholds=thr)
print(f'Training samples: {len(X_tr)}')

# Calculate scale_pos_weight
pos = int((y_tr==1).sum())
neg = int((y_tr==0).sum())
spw = float(neg/max(pos, 1))

# Initialize and train CatBoost
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

val_size = int(len(X_tr) * 0.1)
train_size = len(X_tr) - val_size
train_indices = np.arange(train_size)
val_indices = np.arange(train_size, len(X_tr))

print('Training...')
clf.fit(
    X_tr[train_indices], 
    y_tr[train_indices],
    eval_set=(X_tr[val_indices], y_tr[val_indices])
)

# Save model
model_dict = {'model': clf}
with open('model.pkl', 'wb') as f:
    pickle.dump(model_dict, f)

print(f'Model saved to model.pkl')
print(f'Best iteration: {clf.get_best_iteration()}')
