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
    raise ImportError("PyTorch is required for TCN model. Install with: pip install torch")

from sklearn.preprocessing import StandardScaler

HIST_DAYS=7; FUTURE_DAYS=14
FEATURES = ["sea_level", "sea_level_3d_mean", "sea_level_7d_mean", "sea_level_std"]

class Chomp1d(nn.Module):
    """Causal padding - removes right padding from convolution output"""
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size > 0 else x

class TemporalBlock(nn.Module):
    """Temporal Convolutional Block with dilated convolutions and causal padding"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)  # Causal padding - remove right padding
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCNModel(nn.Module):
    """Temporal Convolutional Network for sequence prediction"""
    def __init__(self, num_inputs=4, num_channels=[64, 128, 64], kernel_size=3, dropout=0.3, output_size=14):
        super(TCNModel, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear1 = nn.Linear(num_channels[-1], num_channels[-1] // 2)
        self.bn = nn.BatchNorm1d(num_channels[-1] // 2)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(num_channels[-1] // 2, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Input shape: (batch, seq_len, features) -> (batch, features, seq_len)
        x = x.transpose(1, 2)
        y = self.network(x)
        # Take last timestep: (batch, channels, seq_len) -> (batch, channels)
        y = y[:, :, -1]
        # Fully connected layers
        y = self.linear1(y)
        y = self.bn(y)
        y = torch.relu(y)
        y = self.dropout(y)
        y = self.linear2(y)
        return self.sigmoid(y)

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
    # Enhanced feature engineering
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
            # For TCN, keep sequence shape (7, 4) for convolutional layers
            X.append(hist_block.values)
            meta.append({"station": stn,
                         "hist_start": grp.loc[i, "date"],
                         "future_start": grp.loc[i+HIST_DAYS, "date"]})
            if use_labels:
                fut = grp.loc[i+HIST_DAYS:i+HIST_DAYS+FUTURE_DAYS-1, "flood"]
                y.append(fut.values)
    X = np.array(X)  # Shape: (n_samples, 7, 4)
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
            
            saved_params = model_dict.get('model_params', {})
            tcn_model = model_dict['model_class'](
                num_inputs=saved_params.get('num_inputs', len(FEATURES)),
                num_channels=saved_params.get('num_channels', [64, 128, 64]),
                kernel_size=saved_params.get('kernel_size', 3),
                dropout=saved_params.get('dropout', 0.3),
                output_size=saved_params.get('output_size', 14)
            )
            tcn_model.load_state_dict(model_dict['model_state_dict'], strict=False)
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

            # Initialize TCN model
            tcn_model = TCNModel(num_inputs=len(FEATURES), num_channels=[64, 128, 64], 
                                kernel_size=3, dropout=0.3, output_size=14)

            criterion = nn.BCELoss()
            optimizer = optim.Adam(tcn_model.parameters(), lr=0.001, weight_decay=1e-5)
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
                tcn_model.train()
                train_loss = 0
                for batch_x, batch_y in train_loader:
                    optimizer.zero_grad()
                    outputs = tcn_model(batch_x)
                    loss = criterion(outputs, batch_y)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(tcn_model.parameters(), max_norm=1.0)
                    optimizer.step()
                    train_loss += loss.item()
                
                tcn_model.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_y in val_loader:
                        outputs = tcn_model(batch_x)
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

            # Save model for future use
            model_dict = {
                'model_state_dict': tcn_model.state_dict(),
                'model_class': TCNModel,
                'model_params': {
                    'num_inputs': len(FEATURES),
                    'num_channels': [64, 128, 64],
                    'kernel_size': 3,
                    'dropout': 0.3,
                    'output_size': 14
                },
                'scaler': scaler,
                'features': FEATURES,
                'hist_days': HIST_DAYS,
                'future_days': FUTURE_DAYS
            }
            with open('model.pkl', 'wb') as f:
                pickle.dump(model_dict, f)

        # Build test windows and align to index by key
        X_te, _, meta_te = build_windows(daily_te, daily_te["station_name"].unique().tolist(), use_labels=False)
        
        if len(X_te)==0:
            raise RuntimeError("No test windows could be built from provided test_hourly/test_index. Check dates.")

        # Normalize test data
        X_te_reshaped = X_te.reshape(-1, len(FEATURES))
        X_te_scaled = scaler.transform(X_te_reshaped).reshape(X_te.shape)
        X_te_tensor = torch.FloatTensor(X_te_scaled)

        # Predict
        tcn_model.eval()
        with torch.no_grad():
            y_pred_probs = tcn_model(X_te_tensor).numpy()
        # Improved aggregation: weighted average + max
        weights = np.linspace(0.5, 1.5, 14)
        weighted_mean = np.average(y_pred_probs, axis=1, weights=weights)
        max_prob = y_pred_probs.max(axis=1)
        probs = 0.6 * weighted_mean + 0.4 * max_prob

        # Convert dates for consistent key matching
        meta_te["hist_start"] = pd.to_datetime(meta_te["hist_start"])
        meta_te["future_start"] = pd.to_datetime(meta_te["future_start"])
        meta_te["key"] = meta_te["station"].astype(str) + "|" + meta_te["hist_start"].astype(str) + "|" + meta_te["future_start"].astype(str)

        index["hist_start"] = pd.to_datetime(index["hist_start"])
        index["future_start"] = pd.to_datetime(index["future_start"])
        index["key"] = index["station_name"].astype(str) + "|" + index["hist_start"].astype(str) + "|" + index["future_start"].astype(str)

        pred_df = pd.DataFrame({"key": meta_te["key"], "y_prob": probs})

        out = index.merge(pred_df, on="key", how="left")[["id","y_prob"]]
        out["y_prob"] = out["y_prob"].fillna(0.5)
        
        # Ensure all IDs from index are present
        if len(out) != len(index):
            print(f"Warning: Only {len(out)} predictions for {len(index)} test cases", flush=True)
            out = index[["id"]].merge(out, on="id", how="left")
            out["y_prob"] = out["y_prob"].fillna(0.5)
        
        out.to_csv(args.predictions_out, index=False)
        print(f"Wrote {args.predictions_out} with {len(out)} predictions", flush=True)
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
