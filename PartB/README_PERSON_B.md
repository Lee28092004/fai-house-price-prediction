# Person B — Optimization, Evaluation, Visualization, and Demo

This folder contains the work for **Person B** in Part 2 of the assignment.

## Responsibilities Completed
- Hyperparameter tuning
- Model evaluation
- Visualization
- Demo readiness
- Final output saving

## Files
- `tuning_and_visualization.py`  
  Tunes Random Forest and XGBoost, evaluates both, generates plots, and saves outputs.

- `predict_demo.py`  
  Loads the best tuned model and predicts one sample house price from `X_test.csv`.

## Evaluation Metrics Used
- MAE
- RMSE
- R²
- MAPE

## Plots Generated
- Actual vs Predicted Scatter Plot
- Actual vs Predicted Line Plot
- Residual Plot
- Residual Distribution Histogram
- Feature Importance Plot
- Correlation Heatmap
- Model Comparison Bar Charts

## Output Folders
- `outputs/`  
  Stores metrics, predictions, summary JSON, and plots

- `models/`  
  Stores the best trained model files

## How to Run
```bash
python PartB/tuning_and_visualization.py
python PartB/predict_demo.py