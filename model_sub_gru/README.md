# GRU Flood Prediction Model

## Overview

This repository contains a deep learning model for predicting coastal flood events using Gated Recurrent Unit (GRU) neural networks. The model takes 7 days of historical sea level data as input and predicts flood probabilities for the next 14 days.

## Model Architecture

The model implements a multi-layer GRU architecture:

- **Input Layer**: 7 timesteps × 4 features (sea_level, 3-day rolling mean, 7-day rolling mean, 7-day std)
- **GRU Layers**: 3 layers with 128 hidden units each, bidirectional, dropout=0.3
- **Attention Mechanism**: Multi-head attention with 4 heads to focus on important timesteps
- **Fully Connected Layers**: 256 → 128 (with batch normalization and dropout) → 64 → 14 outputs
- **Output Layer**: 14 sigmoid units (one per day) predicting flood probability

### Key Features

- **Bidirectional GRU**: Captures patterns in both forward and backward directions
- **Multi-head Attention**: Focuses on important timesteps in the sequence
- **Feature Engineering**: Multiple rolling window features (3-day mean, 7-day mean, 7-day std)
- **Regularization**: Batch normalization and dropout prevent overfitting

## Pipeline Description

### 1. Data Preprocessing

#### Daily Aggregation
- Aggregates hourly sea level data to daily level (mean sea level per day)
- Computes daily maximum sea level (for flood threshold calculation)
- Preserves station metadata (latitude, longitude)

#### Feature Engineering
- **Rolling Means**: Computes 3-day and 7-day rolling means of sea level
- **Rolling Standard Deviation**: Computes 7-day rolling standard deviation
- **Flood Threshold**: Per-station threshold = mean + 1.5 × standard deviation
- **Flood Labels**: Binary labels (1 if daily max > threshold, else 0)

#### Window Construction
- **Historical Window**: 7 consecutive days of features (4 features × 7 days = 28 values)
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
- Aggregates to single probability using **weighted combination**:
  - 60% weighted mean (more weight to later days)
  - 40% maximum probability across all 14 days

## File Structure

```
model_sub_gru/
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
5. Train GRU with early stopping
6. Save best model based on validation loss

## Key Design Decisions

1. **Sequence-based Input**: Using GRU instead of flattened features preserves temporal structure
2. **Bidirectional GRU**: Captures patterns in both directions of the sequence
3. **Attention Mechanism**: Allows the model to focus on important timesteps
4. **Multi-day Output**: Predicting 14 days simultaneously enables capturing dependencies between days
5. **Rolling Features**: 3-day and 7-day means capture short-term trends
6. **Aggregation Strategy**: Weighted combination emphasizes worst-case scenarios while considering temporal patterns
7. **Regularization**: Dropout and batch normalization prevent overfitting to training data

## GRU Advantages

1. **Efficiency**: GRU typically trains faster than LSTM while maintaining similar performance
2. **Memory**: Simpler gating mechanism (2 gates vs 3 in LSTM) reduces computational overhead
3. **Gradient Flow**: Better gradient flow compared to vanilla RNNs
4. **Bidirectional Processing**: Captures both forward and backward temporal dependencies
5. **Attention**: Multi-head attention mechanism focuses on relevant timesteps

## Limitations and Future Improvements

### Current Limitations
- Fixed window size (7 days history, 14 days future)
- Single aggregation strategy (weighted combination)
- No consideration of spatial dependencies between stations

### Potential Improvements
1. **Variable Window Size**: Adaptive window sizes based on station characteristics
2. **Spatial Attention**: Incorporate information from nearby stations
3. **Multi-station Models**: Share information across geographically nearby stations
4. **Ensemble Methods**: Combine multiple models for robustness
5. **Hyperparameter Tuning**: Grid search or Bayesian optimization for optimal architecture
6. **Transfer Learning**: Pre-train on multiple stations and fine-tune per station
