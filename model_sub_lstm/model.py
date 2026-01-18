#!/usr/bin/env python3
"""
Participant code-submission baseline.
Usage (called by ingestion):
python -m model --train_hourly <csv> --test_hourly <csv> --test_index <csv> --predictions_out <csv>
"""
import argparse, pandas as pd, numpy as np, pickle, os
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False
    raise ImportError("PyTorch is required for LSTM model. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler

HIST_DAYS=7; FUTURE_DAYS=14
FEATURES = ["sea_level", "sea_level_3d_mean", "sea_level_7d_mean", "sea_level_std"]

class ImprovedLSTMModel(nn.Module):
    def __init__(self, input_size=3, hidden_size=128, num_layers=3, dropout=0.3, output_size=14):
        super(ImprovedLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Bidirectional LSTM with more capacity
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size * 2, num_heads=4, dropout=dropout, batch_first=True)
        
        # Additional fully connected layers with more capacity
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size // 2)
        self.fc3 = nn.Linear(hidden_size // 2, output_size)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # LSTM forward pass (bidirectional gives 2x hidden_size)
        lstm_out, _ = self.lstm(x)  # Shape: (batch, seq_len, hidden_size * 2)
        
        # Attention mechanism
        attn_out, _ = self.attention(lstm_out, lstm_out, lstm_out)  # Shape: (batch, seq_len, hidden_size * 2)
        
        # Combine last output with attended output
        last_lstm = lstm_out[:, -1, :]  # Last LSTM output
        last_attn = attn_out[:, -1, :]  # Last attended output
        combined = (last_lstm + last_attn) / 2  # Average combination
        
        # Fully connected layers with batch norm and dropout
        out = self.fc1(combined)
        out = self.bn1(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = torch.relu(out)
        out = self.dropout(out)
        
        out = self.fc3(out)
        
        return self.sigmoid(out)

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
            # For LSTM, we keep the sequence shape (7, 3) instead of flattening
            X.append(hist_block.values)
            meta.append({"station": stn,
                         "hist_start": grp.loc[i, "date"],
                         "future_start": grp.loc[i+HIST_DAYS, "date"]})
            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"]
                # For training, we use all 14 days as target
                y.append(fut.values)
    X = np.array(X)  # Shape: (n_samples, 7, 3)
    return X, (np.array(y) if use_labels else None), pd.DataFrame(meta)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_hourly", required=True)
    ap.add_argument("--test_hourly", required=True)
    ap.add_argument("--test_index", required=True)
    ap.add_argument("--predictions_out", required=True)
    args = ap.parse_args()

    try:
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
            
            # Handle potential architecture mismatch - load with compatible params
            saved_params = model_dict.get('model_params', {})
            # Use saved params if available, otherwise use defaults
            lstm_model = model_dict['model_class'](
                input_size=saved_params.get('input_size', len(FEATURES)),
                hidden_size=saved_params.get('hidden_size', 128),
                num_layers=saved_params.get('num_layers', 3),
                dropout=saved_params.get('dropout', 0.3),
                output_size=saved_params.get('output_size', 14)
            )
            lstm_model.load_state_dict(model_dict['model_state_dict'], strict=False)
            scaler = model_dict['scaler']
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

            # Normalize features
            X_tr_reshaped = X_tr.reshape(-1, len(FEATURES))
            scaler = StandardScaler()
            X_tr_scaled = scaler.fit_transform(X_tr_reshaped).reshape(X_tr.shape)

            # Convert to PyTorch tensors
            X_tr_tensor = torch.FloatTensor(X_tr_scaled)
            y_tr_tensor = torch.FloatTensor(y_tr)

            # Initialize LSTM model with improved architecture
            lstm_model = ImprovedLSTMModel(input_size=len(FEATURES), hidden_size=128, num_layers=3, dropout=0.3, output_size=14)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

            # Create data loaders
            train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
            val_size = int(len(X_tr_tensor) * 0.1)
            train_size = len(X_tr_tensor) - val_size
            train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
            train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
            val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)

            # Training with early stopping
            num_epochs = 20
            best_val_loss = float('inf')
            patience = 5
            patience_counter = 0

            for epoch in range(num_epochs):
                lstm_model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = lstm_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(lstm_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                lstm_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = lstm_model(batch_x)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item()
                
                avg_val_loss = val_loss / len(val_loader)
                scheduler.step(avg_val_loss)
                
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= patience:
                    break

        # Build test windows and align to index by key
        X_te, _, meta_te = build_windows(daily_te, daily_te["station_name"].unique().tolist(), use_labels=False)
        
        if len(X_te)==0:
            raise RuntimeError("No test windows could be built from provided test_hourly/test_index. Check dates.")

        # Normalize test data
        X_te_reshaped = X_te.reshape(-1, len(FEATURES))
        X_te_scaled = scaler.transform(X_te_reshaped).reshape(X_te.shape)
        X_te_tensor = torch.FloatTensor(X_te_scaled)

        # Predict
        lstm_model.eval()
        with torch.no_grad():
            y_pred_probs = lstm_model(X_te_tensor).numpy()
        # Improved aggregation: weighted average (more weight to later days) + max
        weights = np.linspace(0.5, 1.5, 14)  # Increasing weights for later days
        weighted_mean = np.average(y_pred_probs, axis=1, weights=weights)
        max_prob = y_pred_probs.max(axis=1)
        # Combine weighted mean (60%) and max (40%) for better prediction
        probs = 0.6 * weighted_mean + 0.4 * max_prob

        meta_te["key"] = meta_te["station"].astype(str) + "|" + meta_te["hist_start"].astype(str) + "|" + meta_te["future_start"].astype(str)

        index["hist_start"] = pd.to_datetime(index["hist_start"])
        index["future_start"] = pd.to_datetime(index["future_start"])
        index["key"] = index["station_name"].astype(str) + "|" + index["hist_start"].astype(str) + "|" + index["future_start"].astype(str)

        pred_df = pd.DataFrame({"key": meta_te["key"], "y_prob": probs})

        out = index.merge(pred_df, on="key", how="left")[["id","y_prob"]]
        out["y_prob"] = out["y_prob"].fillna(0.5)
        out.to_csv(args.predictions_out, index=False)
        print(f"Wrote {args.predictions_out}")
    except Exception as e:
        # Ensure predictions.csv is always created, even on error
        print(f"Error occurred: {e}", flush=True)
        import traceback
        traceback.print_exc()
        
        # Create default predictions file
        try:
            index = pd.read_csv(args.test_index)
            out = pd.DataFrame({"id": index["id"], "y_prob": 0.5})
            out.to_csv(args.predictions_out, index=False)
            print(f"Created default predictions.csv with 0.5 probabilities", flush=True)
        except Exception as e2:
            print(f"Failed to create default predictions: {e2}", flush=True)
            raise

if __name__ == "__main__":
    main()
