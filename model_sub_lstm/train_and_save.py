#!/usr/bin/env python3
"""
Script to train the LSTM model and save model.pkl
This script trains on the full training data and saves the trained model.
"""
import pickle
import sys
from pathlib import Path
from model import (
    ImprovedLSTMModel, daily_aggregate, build_windows,
    HIST_DAYS, FUTURE_DAYS, FEATURES
)
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler

def train_and_save(train_csv_path, model_pkl_path):
    """Train model on training data and save to model.pkl"""
    print(f"Loading training data from {train_csv_path}...")
    train = pd.read_csv(train_csv_path)
    print(f"Loaded {len(train)} hourly records")
    
    print("\nPreprocessing: Aggregating to daily level...")
    daily_tr = daily_aggregate(train)
    print(f"Created {len(daily_tr)} daily records")
    
    print("\nComputing flood thresholds...")
    thr = (train.groupby("station_name")["sea_level"]
                .agg(["mean","std"])
                .assign(flood_threshold=lambda x: x["mean"] + 1.5*x["std"])
                [["flood_threshold"]]
                .reset_index())
    
    print("\nBuilding training windows...")
    stn_tr = daily_tr["station_name"].unique().tolist()
    X_tr, y_tr, _ = build_windows(daily_tr, stn_tr, use_labels=True, thresholds=thr)
    print(f"Training samples: {len(X_tr)}")
    print(f"Positive samples: {y_tr.sum()} / {y_tr.size} ({y_tr.mean():.4f})")
    
    print("\nNormalizing features...")
    X_tr_reshaped = X_tr.reshape(-1, len(FEATURES))
    scaler = StandardScaler()
    X_tr_scaled = scaler.fit_transform(X_tr_reshaped).reshape(X_tr.shape)
    
    print("\nConverting to PyTorch tensors...")
    X_tr_tensor = torch.FloatTensor(X_tr_scaled)
    y_tr_tensor = torch.FloatTensor(y_tr)
    
    print("\nInitializing LSTM model...")
    lstm_model = ImprovedLSTMModel(
        input_size=len(FEATURES), 
        hidden_size=64, 
        num_layers=2,
        dropout=0.3,
        output_size=14
    )
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(lstm_model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    print("\nCreating data loaders...")
    train_dataset = TensorDataset(X_tr_tensor, y_tr_tensor)
    val_size = int(len(X_tr_tensor) * 0.1)
    train_size = len(X_tr_tensor) - val_size
    train_subset, val_subset = torch.utils.data.random_split(train_dataset, [train_size, val_size])
    train_loader = DataLoader(train_subset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
    
    print("\nTraining model...")
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
            best_model_state = lstm_model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch+1}/{num_epochs}], Val Loss: {avg_val_loss:.4f}")
        
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            lstm_model.load_state_dict(best_model_state)
            break
    
    print(f"\nTraining complete. Best validation loss: {best_val_loss:.4f}")
    
    print(f"\nSaving model to {model_pkl_path}...")
    model_dict = {
        'model_state_dict': best_model_state if 'best_model_state' in locals() else lstm_model.state_dict(),
        'model_class': ImprovedLSTMModel,
        'model_params': {
            'input_size': len(FEATURES),
            'hidden_size': 64,
            'num_layers': 2,
            'dropout': 0.3,
            'output_size': 14
        },
        'scaler': scaler,
        'features': FEATURES,
        'hist_days': HIST_DAYS,
        'future_days': FUTURE_DAYS
    }
    
    with open(model_pkl_path, 'wb') as f:
        pickle.dump(model_dict, f)
    
    print(f"Model saved successfully to {model_pkl_path}!")

if __name__ == "__main__":
    # Default paths - can be overridden via command line
    if len(sys.argv) > 1:
        train_csv_path = sys.argv[1]
    else:
        train_csv_path = "../processed_data/train_hourly.csv"
    
    if len(sys.argv) > 2:
        model_pkl_path = sys.argv[2]
    else:
        model_pkl_path = "model.pkl"
    
    train_and_save(train_csv_path, model_pkl_path)
