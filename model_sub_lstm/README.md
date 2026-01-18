# LSTM Flood Prediction Model

## Overview

This repository contains a deep learning model for predicting coastal flood events using Long Short-Term Memory (LSTM) neural networks. The model takes 7 days of historical sea level data as input and predicts flood probabilities for the next 14 days.

## Model Architecture

The model implements a multi-layer LSTM architecture:

- **Input Layer**: 7 timesteps × 3 features (sea_level, 3-day rolling mean, 7-day rolling mean)
- **LSTM Layers**: 2 layers with 64 hidden units each, dropout=0.3
- **Fully Connected Layers**: 64 → 32 (with batch normalization and dropout) → 14 outputs
- **Output Layer**: 14 sigmoid units (one per day) predicting flood probability

## Pipeline Description

### 1. Data Preprocessing

#### Daily Aggregation
- Aggregates hourly sea level data to daily level (mean sea level per day)
- Computes daily maximum sea level (for flood threshold calculation)
- Preserves station metadata (latitude, longitude)

#### Feature Engineering
- **Rolling Means**: Computes 3-day and 7-day rolling means of sea level
- **Flood Threshold**: Per-station threshold = mean + 1.5 × standard deviation
- **Flood Labels**: Binary labels (1 if daily max > threshold, else 0)

#### Window Construction
- **Historical Window**: 7 consecutive days of features (3 features × 7 days = 21 values)
- **Target Window**: 14 consecutive days of flood labels
- **Sliding Window**: Creates overlapping windows for training/prediction

### 2. Model Development

#### Data Normalization
- **StandardScaler**: Normalizes features (mean=0, std=1) across all timesteps
- Fitted on training data only, applied to test data

#### Model Training
- **Optimizer**: Adam with learning rate 0.001, weight decay 1e-5
- **Loss Function**: Binary Cross-Entropy (BCELoss)
- **Learning Rate Scheduling**: ReduceLROnPlateau (factor=0.5, patience=3)
- **Early Stopping**: Stops if validation loss doesn't improve for 5 epochs
- **Gradient Clipping**: Max norm = 1.0 (prevents exploding gradients)
- **Batch Size**: 128 samples
- **Validation Split**: 10% of training data
- **Max Epochs**: 20

### 3. Model Evaluation

#### Prediction Aggregation
- Model outputs 14-day probabilities (one per day)
- Aggregates to single probability using **maximum** across all 14 days
- This captures worst-case scenarios (highest flood risk in the window)

## File Structure

```
model_sub_lstm/
├── model.py          # Full training and evaluation pipeline
├── model.pkl         # Trained model weights and scaler
├── Requirements.txt  # Python package dependencies
└── README.md         # This file
```

## Requirements

### Python Version
Python 3.8 or higher

### Dependencies
```
torch>=2.0.0          # PyTorch for deep learning
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computations
scikit-learn>=1.3.0   # StandardScaler and metrics
```

Install dependencies:
```bash
pip install -r Requirements.txt
```

## Usage

### Training and Prediction

To train the model and generate predictions:

```bash
python model.py \
    --train_hourly train_hourly.csv \
    --test_hourly test_hourly.csv \
    --test_index test_index.csv \
    --predictions_out predictions.csv
```

### Expected Input Format

#### train_hourly.csv / test_hourly.csv
Required columns:
- `time`: Timestamp (datetime format)
- `station_name`: Station identifier (string)
- `sea_level`: Sea level measurement (float)
- `latitude`: Station latitude (float)
- `longitude`: Station longitude (float)

#### test_index.csv
Required columns:
- `id`: Unique prediction identifier
- `station_name`: Station identifier
- `hist_start`: Start date of historical window (datetime)
- `future_start`: Start date of prediction window (datetime)

### Expected Output Format

#### predictions.csv
- `id`: Prediction identifier (matches test_index.csv)
- `y_prob`: Flood probability (float, 0-1)

## Model Loading

To load a trained model:

```python
import pickle
import torch

# Load model dictionary
with open('model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# Reconstruct model
model_class = model_dict['model_class']
model = model_class(**model_dict['model_params'])
model.load_state_dict(model_dict['model_state_dict'])

# Load scaler
scaler = model_dict['scaler']

# Use for prediction
model.eval()
# ... preprocessing ...
# ... prediction ...
```

## Reproducibility

### Training Process
1. Load and preprocess training data
2. Build sliding windows (7-day history → 14-day future)
3. Normalize features using StandardScaler
4. Split into train/validation (90/10)
5. Train LSTM with early stopping
6. Save best model based on validation loss

## Key Design Decisions

1. **Sequence-based Input**: Using LSTM instead of flattened features preserves temporal structure
2. **Multi-day Output**: Predicting 14 days simultaneously enables capturing dependencies between days
3. **Rolling Features**: 3-day and 7-day means capture short-term trends
4. **Aggregation Strategy**: Using max probability across 14 days emphasizes worst-case scenarios
5. **Regularization**: Dropout and batch normalization prevent overfitting to training data

## Limitations and Future Improvements

### Current Limitations
- Fixed window size (7 days history, 14 days future)
- Single aggregation strategy (max probability)
- No consideration of spatial dependencies between stations

### Potential Improvements
1. **Bidirectional LSTM**: Capture patterns in both directions
2. **Attention Mechanism**: Focus on important timesteps
3. **Multi-station Models**: Share information across geographically nearby stations
4. **Ensemble Methods**: Combine multiple models for robustness
5. **Adaptive Thresholds**: Station-specific thresholds based on local patterns
