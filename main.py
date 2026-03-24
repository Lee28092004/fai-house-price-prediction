"""
Main Pipeline — House Price Prediction
Full pipeline combining Person A and Person B work.

Runs all steps in order:
    Step 0 — detection.py                : audit raw data for issues
    Step 1 — data_cleaning.py            : clean raw data → houses_clean.csv
    Step 2 — feature_selection.py        : statistically select features → feature_selection_result.json
    Step 3 — feature_engineering.py      : engineer + encode features → train/test splits
    Step 4 — model_building.py           : build and baseline-train RF + XGBoost models
    Step 5 — tuning_and_visualization.py : tune models, evaluate, generate all plots

Run from the project root:
    python main.py

Expected folder structure:
    project/
    ├── main.py
    ├── detection.py
    ├── data_cleaning.py
    ├── feature_selection.py
    ├── feature_engineering.py
    ├── model_building.py
    ├── tuning_and_visualization.py
    ├── data/
    │   └── houses.csv
    ├── model/              ← created automatically
    └── outputs/            ← created automatically by Step 5
        └── plots/
"""

import os
import subprocess
import sys

STEPS = [
    ("Step 0 — Data Detection",           "detection.py"),
    ("Step 1 — Data Cleaning",             "data_cleaning.py"),
    ("Step 2 — Feature Selection",         "feature_selection.py"),
    ("Step 3 — Feature Engineering",       "feature_engineering.py"),
    ("Step 4 — Model Building",            "model_building.py"),
    ("Step 5 — Tuning & Visualization",    "tuning_and_visualization.py"),
]

DIVIDER = "=" * 60


def ask_to_continue(next_label):
    while True:
        answer = input(f"\n  Ready to run {next_label}? (yes / no): ").strip().lower()
        if answer in ('yes', 'y'):
            return
        elif answer in ('no', 'n'):
            print("  Pipeline paused. Re-run main.py to start again.")
            sys.exit(0)
        else:
            print("  Please type yes or no.")


def run_step(label, script):
    print(f"\n{DIVIDER}")
    print(f"  {label}")
    print(f"  Script: {script}")
    print(DIVIDER)

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    if result.returncode != 0:
        print(f"\n  FAILED: {script} exited with code {result.returncode}")
        print(f"  Pipeline stopped. Fix the error above and re-run main.py.")
        sys.exit(result.returncode)

    print(f"\n  {label} — completed successfully.")


if __name__ == "__main__":
    # Ensure required directories exist before any script runs
    os.makedirs("data", exist_ok=True)
    os.makedirs("model", exist_ok=True)

    # Confirm raw data is present before starting
    if not os.path.exists(os.path.join("data", "houses.csv")):
        print("ERROR: data/houses.csv not found.")
        print("Place the raw dataset in the data/ folder and re-run.")
        sys.exit(1)

    print(DIVIDER)
    print("  HOUSE PRICE PREDICTION — PERSON A PIPELINE")
    print(DIVIDER)
    print(f"  Running {len(STEPS)} steps in sequence.")
    print(f"  Any failure will stop the pipeline immediately.")

    for i, (label, script) in enumerate(STEPS):
        if i > 0:
            ask_to_continue(label)
        run_step(label, script)

    print(f"\n{DIVIDER}")
    print("  ALL STEPS COMPLETE")
    print(DIVIDER)
    print("""
  Outputs produced:
    data/houses_clean.csv              — cleaned dataset
    data/feature_selection_result.json — selected features
    data/X_train.csv                   — training features
    data/X_test.csv                    — test features
    data/y_train.csv                   — training targets
    data/y_test.csv                    — test targets
    data/houses_engineered.csv         — full engineered dataset
    model/rf_model_baseline.pkl        — baseline Random Forest
    model/xgb_model_baseline.json      — baseline XGBoost
    model/best_random_forest.pkl       — tuned Random Forest
    model/best_xgboost.json            — tuned XGBoost
    outputs/model_metrics.csv          — evaluation metrics
    outputs/summary.json               — tuning summary
    outputs/rf_predictions.csv         — RF predictions vs actuals
    outputs/xgb_predictions.csv        — XGBoost predictions vs actuals
    outputs/plots/                     — all visualisation plots
""")