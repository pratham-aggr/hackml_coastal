# ML Challenge 2
Important note for the baseline model:
1. The example submission uses 'alternative' thresholds as placeholders for the model.

Please refer to the .mat file for the flooding thresholds ('Seed Coastal Stations Thresholds.mat') in this repository.

2. Given 12 coastal stations, 9 stations are fixed for training and 3 for testing, which aligns with how the model will be processed (during ingestion) and scored.

Therefore, the results for the evaluation metrics of the baseline model may vary between the local machine and Codabench (which has predetermined training and test sets for out-of-distribution modelling). This type of evaluation will be applied during the final phase on the hidden dataset.

Refer to the zipped submission files to understand the expectations:
- Example 1: model_submission.zip contains only the model.py, which is the baseline model in this case.
- Example 2: model_submission-2.zip contains ALL files as outlined under the "Expected Submission Files" on Codabench. These include model.py, model.pkl, requirements.txt, and README.md
