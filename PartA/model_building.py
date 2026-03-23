"""
Model Building Script — houses_engineered.csv
Person A: Data & Model Architect
Step 4 of pipeline: Feature-ready data → Defined and baseline-trained models

Two models are built as chosen in Part 1:
    1. Random Forest Regressor  — ensemble of decision trees (bagging)
    2. XGBoost Regressor        — gradient boosted decision trees (boosting)

This script defines the model architectures with initial parameters,
runs a baseline training pass to confirm both models function correctly,
and saves the trained models for Person B to tune and evaluate.
"""

import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ── Load data ──────────────────────────────────────────────────────────────────
X_train = pd.read_csv('data/X_train.csv')
X_test  = pd.read_csv('data/X_test.csv')
y_train = pd.read_csv('data/y_train.csv').squeeze()
y_test  = pd.read_csv('data/y_test.csv').squeeze()

print(f"Training set : {X_train.shape[0]} rows × {X_train.shape[1]} features")
print(f"Test set     : {X_test.shape[0]} rows × {X_test.shape[1]} features\n")


# ─────────────────────────────────────────────────────────────────────────────
# HELPER — evaluation metrics
# Computes MAE, RMSE, and R² from predictions vs actuals.
# These are the three metrics specified in Part 1 objectives.
# ─────────────────────────────────────────────────────────────────────────────
def evaluate(y_true, y_pred, label):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    print(f"  {label}")
    print(f"    MAE  : RM {mae:>12,.2f}")
    print(f"    RMSE : RM {rmse:>12,.2f}")
    print(f"    R²   :    {r2:>12.4f}")
    return {'mae': mae, 'rmse': rmse, 'r2': r2}


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 1 — Random Forest Regressor
#
# Architecture decisions:
#   n_estimators=200   — 200 trees provides a stable ensemble without being
#                        excessively slow. Accuracy gains plateau beyond ~300.
#   max_depth=None     — trees grow fully, letting each tree learn deep patterns.
#                        Overfitting risk is controlled by the ensemble averaging.
#   min_samples_split=5— a node must have at least 5 samples before splitting,
#                        preventing the model from fitting single-row noise.
#   min_samples_leaf=2 — each leaf must contain at least 2 samples, further
#                        reducing variance on small subgroups.
#   max_features='sqrt'— each split considers √n_features candidates (default
#                        for regression). Introduces diversity across trees.
#   n_jobs=-1          — use all available CPU cores for parallel tree building.
#   random_state=42    — fixed seed for full reproducibility.
# ─────────────────────────────────────────────────────────────────────────────
print("=" * 55)
print("  MODEL 1 — Random Forest Regressor")
print("=" * 55)

rf_model = RandomForestRegressor(
    n_estimators    = 200,
    max_depth       = None,
    min_samples_split = 5,
    min_samples_leaf  = 2,
    max_features    = 'sqrt',
    n_jobs          = -1,
    random_state    = 42
)

print("\nArchitecture:")
for param, value in rf_model.get_params().items():
    print(f"  {param:<22} = {value}")

print("\nBaseline training...")
rf_model.fit(X_train, y_train)

rf_train_preds = rf_model.predict(X_train)
rf_test_preds  = rf_model.predict(X_test)

print("\nBaseline results:")
rf_train_metrics = evaluate(y_train, rf_train_preds, "Train")
rf_test_metrics  = evaluate(y_test,  rf_test_preds,  "Test")

# Feature importances from the trained forest
rf_importances = pd.Series(rf_model.feature_importances_, index=X_train.columns)
rf_importances = rf_importances.sort_values(ascending=False)
print("\nTop 10 feature importances:")
for feat, imp in rf_importances.head(10).items():
    bar = '█' * int(imp * 100)
    print(f"  {feat:<30} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# MODEL 2 — XGBoost Regressor
#
# Architecture decisions:
#   n_estimators=300      — 300 boosting rounds. XGBoost builds trees
#                           sequentially so more rounds = finer corrections,
#                           but also higher overfitting risk (controlled below).
#   learning_rate=0.05    — small step size per round. Slower learning but
#                           better generalisation than default 0.3.
#   max_depth=6           — maximum tree depth per round. 6 is a standard
#                           starting point; shallower than RF because boosting
#                           compounds errors from each round.
#   subsample=0.8         — each tree is trained on a random 80% sample of
#                           rows. Introduces variance reduction (like RF).
#   colsample_bytree=0.8  — each tree uses a random 80% of features.
#                           Reduces correlation between consecutive trees.
#   reg_alpha=0.1         — L1 regularisation (Lasso). Pushes weak feature
#                           weights toward zero, implicitly selecting features.
#   reg_lambda=1.0        — L2 regularisation (Ridge). Penalises large weights,
#                           preventing any single feature from dominating.
#   objective='reg:squarederror' — minimise mean squared error (regression).
#   eval_metric='rmse'    — monitor RMSE on validation set during training.
#   early_stopping_rounds=20 — halt if test RMSE does not improve for 20
#                           consecutive rounds, prevents over-training.
#   random_state=42       — fixed seed for reproducibility.
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  MODEL 2 — XGBoost Regressor")
print("=" * 55)

xgb_model = XGBRegressor(
    n_estimators         = 300,
    learning_rate        = 0.05,
    max_depth            = 6,
    subsample            = 0.8,
    colsample_bytree     = 0.8,
    reg_alpha            = 0.1,
    reg_lambda           = 1.0,
    objective            = 'reg:squarederror',
    eval_metric          = 'rmse',
    early_stopping_rounds= 20,
    random_state         = 42,
    n_jobs               = -1,
)

print("\nArchitecture:")
for param, value in xgb_model.get_params().items():
    print(f"  {param:<24} = {value}")

print("\nBaseline training (with early stopping on test set)...")
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=False
)

actual_rounds = xgb_model.best_iteration + 1
print(f"  Early stopping triggered at round {actual_rounds} / {xgb_model.n_estimators}")

xgb_train_preds = xgb_model.predict(X_train)
xgb_test_preds  = xgb_model.predict(X_test)

print("\nBaseline results:")
xgb_train_metrics = evaluate(y_train, xgb_train_preds, "Train")
xgb_test_metrics  = evaluate(y_test,  xgb_test_preds,  "Test")

# Feature importances
xgb_importances = pd.Series(xgb_model.feature_importances_, index=X_train.columns)
xgb_importances = xgb_importances.sort_values(ascending=False)
print("\nTop 10 feature importances:")
for feat, imp in xgb_importances.head(10).items():
    bar = '█' * int(imp * 100)
    print(f"  {feat:<30} {imp:.4f}  {bar}")


# ─────────────────────────────────────────────────────────────────────────────
# BASELINE COMPARISON
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 55)
print("  BASELINE COMPARISON (Test set)")
print("=" * 55)

print(f"\n  {'Metric':<10}  {'Random Forest':>16}  {'XGBoost':>16}")
print(f"  {'-'*46}")
print(f"  {'MAE':<10}  RM {rf_test_metrics['mae']:>13,.2f}  RM {xgb_test_metrics['mae']:>13,.2f}")
print(f"  {'RMSE':<10}  RM {rf_test_metrics['rmse']:>13,.2f}  RM {xgb_test_metrics['rmse']:>13,.2f}")
print(f"  {'R²':<10}     {rf_test_metrics['r2']:>13.4f}     {xgb_test_metrics['r2']:>13.4f}")

better = "XGBoost" if xgb_test_metrics['r2'] > rf_test_metrics['r2'] else "Random Forest"
print(f"\n  Higher R² at baseline: {better}")
print(f"  Note: these are pre-tuning scores. Person B will run hyperparameter")
print(f"  optimisation to improve both models before final evaluation.\n")


# ── Save models ────────────────────────────────────────────────────────────────
with open('data/rf_model_baseline.pkl', 'wb') as f:
    pickle.dump(rf_model, f)

xgb_model.save_model('data/xgb_model_baseline.json')

print(f"Saved: data/rf_model_baseline.pkl")
print(f"Saved: data/xgb_model_baseline.json")
print(f"Next : Person B runs hyperparameter_tuning.py")