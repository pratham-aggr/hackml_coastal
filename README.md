# Coastal Flood Prediction Model

Contents
- `model.py` : Lightweight model code. Includes data preprocessing helpers, training/prediction entrypoint used by the ingestion program, and a `--demo` mode for synthetic training/evaluation.
- `model.pkl` : (generated) serialized trained RandomForest model (demo).
- `requirements.txt` : pinned Python dependencies.

Reproducible pipeline

1) Install dependencies (use a virtual environment):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

2) Demo training (quick, synthetic dataset):

```powershell
python model.py --demo
# this trains a RandomForest on a synthetic dataset, prints test metrics,
# and writes `model.pkl` to the repo root.
```

3) Full ingestion usage (when you have real CSV inputs):

```powershell
python model.py \
  --train_hourly path\to\train_hourly.csv \
  --test_hourly path\to\test_hourly.csv \
  --test_index path\to\test_index.csv \
  --predictions_out out_predictions.csv
```

Notes about `model.py`:
- Aggregates hourly sea level to daily mean and daily max.
- Computes station-specific thresholds (mean + 1.5*std) from training hourly data.
- Builds 7-day history features and predicts whether any flood occurs in the next 14 days.
- The `--demo` mode performs a synthetic training run and saves `model.pkl`.

Evaluation
- The demo prints test accuracy and ROC AUC for the synthetic data split.

Packaging
- The `submission.zip` artifact contains the files required for submission: `model.py`, `requirements.txt`, `model.pkl`, `README.md`.

# iHARP ML Challenge 2 - 'Predicting Coastal Flooding Events.'

# This is a repository for the Year 2 HDR ML Challenge themed 'Modelling Out Of Distribution.'

***Important note for the baseline model:***
1. The example submission uses 'alternative' thresholds as placeholders for the model.
Please refer to the .mat file for the flooding thresholds ('Seed Coastal Stations Thresholds.mat') in this repository.

2. Given 12 coastal stations, 9 stations are fixed for training and 3 for testing, which aligns with how the model will be processed (during ingestion) and scored.

Therefore, the results for the evaluation metrics of the baseline model may vary between the local machine and Codabench (which has predetermined training and test sets for out-of-distribution modelling). This type of evaluation will be applied during the final phase on the hidden dataset.

3. ***Refer to the zipped submission files to understand the expectations:***
- Example 1: model_submission.zip contains only the model.py, which is the baseline model in this case.
- Example 2: model_submission-2.zip contains ALL files as outlined under the "Expected Submission Files" on Codabench. These include model.py, model.pkl, requirements.txt, and README.md.
  
  ***Both types of submissions will be executed appropriately in Codabench***
