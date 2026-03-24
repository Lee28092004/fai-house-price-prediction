# Person B — Optimization, Evaluation, Visualization, and Demo

This folder contains the work completed for **Person B** in Part 2 of the assignment. The focus of this role is model optimization, performance evaluation, results presentation, and demo readiness.

## What this work covers
- Hyperparameter tuning for Random Forest and XGBoost
- Regression evaluation using MAE, RMSE, R², and MAPE
- Professional plots for comparison, residual analysis, and interpretation
- Saved outputs for models, predictions, metrics, and report support files
- A runnable demo script for marker-friendly presentation

## Why these metrics are used
This project predicts **house price**, which is a continuous numeric target. Because the task is **regression**, the correct metrics are:
- MAE
- RMSE
- R²
- MAPE

Precision, Recall, and F1-score are classification metrics, so they are not used for this project.

## Files
- `tuning_and_visualization.py`  
  Tunes Random Forest and XGBoost, evaluates both models, generates all required plots, saves outputs, and exports plot captions plus a demo walkthrough.

- `predict_demo.py`  
  Loads the saved model results, selects one random row from `X_test.csv`, prints the prediction, and presents a step-by-step demo flow from input to output.

## Output folders and files
- `outputs/model_metrics.csv`  
  Table of evaluation metrics for all tuned models.

- `outputs/summary.json`  
  Summary of best hyperparameters, CV score, and test metrics.

- `outputs/rf_predictions.csv` and `outputs/xgb_predictions.csv`  
  Saved actual values, predicted values, and residuals.

- `outputs/plots/`  
  Stores all generated figures.

- `outputs/plot_captions.md`  
  Ready-to-use captions for each plot in the report.

- `outputs/demo_walkthrough.md`  
  Step-by-step explanation for the demonstration section.

- `models/`  
  Stores the saved best trained model files.

## Plots generated
- Actual vs Predicted Scatter Plot
- Actual vs Predicted Line Plot
- Residual Plot
- Residual Distribution Histogram
- Feature Importance Plot
- Correlation Heatmap
- Model Comparison Bar Charts

## Libraries and tools used
- Python 3
- pandas
- numpy
- matplotlib
- scikit-learn
- xgboost (optional; script still runs without it)
- pickle
- json

## How to run
From the project folder, run:

```bash
python PartB/tuning_and_visualization.py
python PartB/predict_demo.py