# CatBoost Flood Prediction Model

## Overview

This repository contains a gradient boosting model for predicting coastal flood events using CatBoost, a high-performance gradient boosting library. The model takes 7 days of historical sea level data as input and predicts flood probabilities for the next 14 days.

## Model Architecture

The model implements CatBoost (Categorical Boosting), a gradient boosting framework:

- **Algorithm**: Gradient Boosting with Categorical Features Support
- **Base Learners**: Decision Trees
- **Depth**: 6 levels
- **Iterations**: 400 trees
- **Learning Rate**: 0.05
- **Loss Function**: Logloss (binary classification)
- **Evaluation Metric**: AUC (Area Under ROC Curve)
- **Early Stopping**: 50 rounds patience

### Key Features

- **Automatic Categorical Handling**: CatBoost's native support for categorical features
- **Overfitting Prevention**: Built-in regularization and early stopping
- **Class Imbalance Handling**: Automatic scale_pos_weight calculation
- **Feature Engineering**: Multiple rolling window features (3-day mean, 7-day mean, 7-day std)

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
- **Feature Vector**: Flattened sequence (28 features) for CatBoost
- **Target**: Binary indicator (1 if any flood in 14-day window, else 0)

### 2. Model Development

#### Training Configuration
- **Iterations**: 400 boosting rounds
- **Depth**: 6 (tree depth)
- **Learning Rate**: 0.05
- **Loss Function**: Logloss (binary cross-entropy)
- **Evaluation Metric**: AUC
- **Early Stopping**: Stops if validation AUC doesn't improve for 50 rounds
- **Class Weighting**: Automatic calculation of scale_pos_weight for imbalanced classes
- **Validation Split**: 10% of training data

#### Model Training
- Train/validation split (90/10)
- Early stopping based on validation AUC
- Automatic hyperparameter optimization
- Built-in overfitting prevention

### 3. Model Evaluation

#### Prediction Process
- Model outputs probability for each test window
- Binary prediction threshold: 0.5
- Handles missing predictions with default 0.5 probability

## File Structure

```
model_sub_catboost/
├── model.py          # Full training and evaluation pipeline
├── model.pkl         # Trained model weights
├── Requirements.txt  # Python package dependencies
└── README.md         # This file
```

## Requirements

### Python Version
Python 3.8 or higher

### Dependencies
```
catboost>=1.2.0       # CatBoost gradient boosting library
pandas>=2.0.0         # Data manipulation
numpy>=1.24.0         # Numerical computations
scikit-learn>=1.3.0   # Utilities and metrics
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

## Generating model.pkl

To generate the `model.pkl` file with a trained model:

1. Install dependencies: `pip install -r Requirements.txt`
2. Ensure training data is available in `../processed_data/train_hourly.csv`
3. Run the model once with training data - it will automatically save `model.pkl` if it doesn't exist
4. Or use the training script: `python train_model.py` (if train_model.py is provided)

The `model.py` includes full training code and will automatically train and save `model.pkl` when:
- `model.pkl` doesn't exist in the current directory
- Training data is provided via `--train_hourly` argument

## Model Loading

To load a trained model:

```python
import pickle

# Load model dictionary
with open('model.pkl', 'rb') as f:
    model_dict = pickle.load(f)

# Get trained model
clf = model_dict['model']

# Use for prediction
probs = clf.predict_proba(X_test)[:, 1]
```

## Reproducibility

### Training Process
1. Load and preprocess training data
2. Build sliding windows (7-day history → 14-day future)
3. Flatten sequences to feature vectors (28 features per window)
4. Calculate class weights for imbalanced data
5. Split into train/validation (90/10)
6. Train CatBoost with early stopping
7. Save best model based on validation AUC

### Model Parameters
- **Iterations**: 400
- **Depth**: 6
- **Learning Rate**: 0.05
- **Early Stopping Rounds**: 50
- **Random Seed**: 42
- **Scale Pos Weight**: Automatically calculated from class distribution

## Key Design Decisions

1. **Flattened Features**: CatBoost works best with flat feature vectors, so sequences are flattened (7 days × 4 features = 28 features)
2. **Binary Target**: Uses max flood indicator across 14-day window (1 if any flood, else 0)
3. **Class Imbalance**: Automatic calculation of scale_pos_weight to handle imbalanced classes
4. **Early Stopping**: Prevents overfitting by stopping when validation AUC plateaus
5. **Feature Engineering**: Multiple rolling windows capture short-term trends and variability

## CatBoost Advantages

1. **Categorical Features**: Native support for categorical variables (if needed)
2. **GPU Acceleration**: Can utilize GPU for faster training (CPU mode used here)
3. **Overfitting Prevention**: Built-in regularization and early stopping
4. **Robustness**: Less prone to overfitting compared to other gradient boosting methods
5. **Interpretability**: Feature importance scores available

## Performance

CatBoost typically achieves:
- **Fast Training**: Efficient implementation with early stopping
- **Good Generalization**: Built-in overfitting prevention
- **High Accuracy**: Competitive performance on tabular data
- **Robust to Outliers**: Less sensitive to outliers than linear models

## Limitations and Future Improvements

### Current Limitations
- Fixed window size (7 days history, 14 days future)
- Binary classification (any flood vs. no flood)
- No consideration of spatial dependencies between stations

### Potential Improvements
1. **Feature Engineering**: Additional temporal features (tides, seasons, etc.)
2. **Multi-station Features**: Incorporate features from nearby stations
3. **Time-based Features**: Day of year, month, seasonal indicators
4. **Ensemble Methods**: Combine multiple CatBoost models or with other algorithms
5. **Hyperparameter Tuning**: Grid search or Bayesian optimization for optimal parameters

## References

- CatBoost Documentation: https://catboost.ai/
- Gradient Boosting Overview: https://en.wikipedia.org/wiki/Gradient_boosting
