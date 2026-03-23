import os
import json
import pickle
import warnings
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from math import sqrt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")
plt.ioff()

XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except ImportError:
    XGB_AVAILABLE = False


# =========================================================
# PATH SETUP
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# =========================================================
# LOAD DATA
# =========================================================
X_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()

print("=" * 60)
print("PERSON B — TUNING, EVALUATION, AND VISUALIZATION")
print("=" * 60)
print(f"X_train shape: {X_train.shape}")
print(f"X_test shape : {X_test.shape}")
print(f"y_train size : {y_train.shape}")
print(f"y_test size  : {y_test.shape}")
print()


# =========================================================
# METRICS
# =========================================================
def mean_absolute_percentage_error_safe(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_regression(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    print(f"{model_name} Results")
    print(f"  MAE  : RM {mae:,.2f}")
    print(f"  RMSE : RM {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print()

    return metrics


# =========================================================
# PLOTTING HELPERS
# =========================================================
def save_actual_vs_predicted_scatter(y_true, y_pred, model_name, filename):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val])
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_actual_vs_predicted_line(y_true, y_pred, model_name, filename, n_points=100):
    plot_df = pd.DataFrame({
        "Actual": np.array(y_true),
        "Predicted": np.array(y_pred)
    }).reset_index(drop=True)

    plot_df = plot_df.iloc[:n_points]

    plt.figure(figsize=(12, 6))
    plt.plot(plot_df.index, plot_df["Actual"], label="Actual")
    plt.plot(plot_df.index, plot_df["Predicted"], label="Predicted")
    plt.xlabel("Test Sample Index")
    plt.ylabel("House Price")
    plt.title(f"Actual vs Predicted Line Graph — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_residual_plot(y_true, y_pred, model_name, filename):
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0)
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_residual_distribution(y_true, y_pred, model_name, filename):
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(8, 6))
    plt.hist(residuals, bins=30)
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.title(f"Residual Distribution — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_feature_importance(model, feature_names, model_name, filename, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"][::-1], importance_df["Importance"][::-1])
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_correlation_heatmap(X, filename):
    corr = X.corr()

    plt.figure(figsize=(14, 10))
    im = plt.imshow(corr, aspect="auto")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_model_comparison_bar(metrics_df, filename_prefix):
    metric_names = ["MAE", "RMSE", "R2", "MAPE"]

    for metric in metric_names:
        plt.figure(figsize=(8, 6))
        plt.bar(metrics_df["Model"], metrics_df[metric])
        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"Model Comparison — {metric}")
        plt.tight_layout()
        plt.savefig(os.path.join(PLOT_DIR, f"{filename_prefix}_{metric.lower()}.png"), dpi=300)
        plt.close()


# =========================================================
# TUNING FUNCTIONS
# =========================================================
def tune_random_forest(X_train, y_train):
    print("=" * 60)
    print("TUNING RANDOM FOREST")
    print("=" * 60)

    rf = RandomForestRegressor(random_state=42, n_jobs=1)

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [None, 10],
        "min_samples_split": [2, 5],
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best RF Parameters:")
    print(grid.best_params_)
    print(f"Best RF CV Score: {grid.best_score_:.4f}")
    print()

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_xgboost(X_train, y_train):
    print("=" * 60)
    print("TUNING XGBOOST")
    print("=" * 60)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1
    )

    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [3, 5],
        "learning_rate": [0.05, 0.1],
    }

    cv = KFold(n_splits=3, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring="r2",
        cv=cv,
        n_jobs=1,
        verbose=1
    )

    grid.fit(X_train, y_train)

    print("Best XGBoost Parameters:")
    print(grid.best_params_)
    print(f"Best XGBoost CV Score: {grid.best_score_:.4f}")
    print()

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================
rf_best, rf_best_params, rf_best_cv = tune_random_forest(X_train, y_train)
rf_best.fit(X_train, y_train)
rf_pred = rf_best.predict(X_test)
rf_metrics = evaluate_regression(y_test, rf_pred, "Random Forest Tuned")


# =========================================================
# TRAIN XGBOOST
# =========================================================
xgb_best = None
xgb_params = None
xgb_cv = None
xgb_pred = None
xgb_metrics = None

if XGB_AVAILABLE:
    xgb_best, xgb_params, xgb_cv = tune_xgboost(X_train, y_train)
    xgb_best.fit(X_train, y_train)
    xgb_pred = xgb_best.predict(X_test)
    xgb_metrics = evaluate_regression(y_test, xgb_pred, "XGBoost Tuned")
else:
    print("XGBoost is not installed. Skipping XGBoost tuning.\n")


# =========================================================
# SAVE METRICS
# =========================================================
all_metrics = [rf_metrics]
if xgb_metrics is not None:
    all_metrics.append(xgb_metrics)

metrics_df = pd.DataFrame(all_metrics)
metrics_df.to_csv(os.path.join(OUTPUT_DIR, "model_metrics.csv"), index=False)

summary = {
    "random_forest": {
        "best_params": rf_best_params,
        "best_cv_score_r2": rf_best_cv,
        "test_metrics": {
            "MAE": rf_metrics["MAE"],
            "RMSE": rf_metrics["RMSE"],
            "R2": rf_metrics["R2"],
            "MAPE": rf_metrics["MAPE"]
        }
    }
}

if xgb_metrics is not None:
    summary["xgboost"] = {
        "best_params": xgb_params,
        "best_cv_score_r2": xgb_cv,
        "test_metrics": {
            "MAE": xgb_metrics["MAE"],
            "RMSE": xgb_metrics["RMSE"],
            "R2": xgb_metrics["R2"],
            "MAPE": xgb_metrics["MAPE"]
        }
    }

with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
    json.dump(summary, f, indent=4)


# =========================================================
# SAVE PREDICTIONS
# =========================================================
rf_pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred,
    "Residual": np.array(y_test) - np.array(rf_pred)
})
rf_pred_df.to_csv(os.path.join(OUTPUT_DIR, "rf_predictions.csv"), index=False)

if xgb_pred is not None:
    xgb_pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": xgb_pred,
        "Residual": np.array(y_test) - np.array(xgb_pred)
    })
    xgb_pred_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_predictions.csv"), index=False)


# =========================================================
# SAVE MODELS
# =========================================================
with open(os.path.join(MODEL_DIR, "best_random_forest.pkl"), "wb") as f:
    pickle.dump(rf_best, f)

if xgb_best is not None:
    xgb_best.save_model(os.path.join(MODEL_DIR, "best_xgboost.json"))


# =========================================================
# GENERATE PLOTS
# =========================================================
print("=" * 60)
print("GENERATING PLOTS")
print("=" * 60)

save_actual_vs_predicted_scatter(
    y_test, rf_pred, "Random Forest", "rf_actual_vs_predicted_scatter.png"
)
save_actual_vs_predicted_line(
    y_test, rf_pred, "Random Forest", "rf_actual_vs_predicted_line.png"
)
save_residual_plot(
    y_test, rf_pred, "Random Forest", "rf_residual_plot.png"
)
save_residual_distribution(
    y_test, rf_pred, "Random Forest", "rf_residual_distribution.png"
)
save_feature_importance(
    rf_best, X_train.columns, "Random Forest", "rf_feature_importance.png"
)

if xgb_pred is not None:
    save_actual_vs_predicted_scatter(
        y_test, xgb_pred, "XGBoost", "xgb_actual_vs_predicted_scatter.png"
    )
    save_actual_vs_predicted_line(
        y_test, xgb_pred, "XGBoost", "xgb_actual_vs_predicted_line.png"
    )
    save_residual_plot(
        y_test, xgb_pred, "XGBoost", "xgb_residual_plot.png"
    )
    save_residual_distribution(
        y_test, xgb_pred, "XGBoost", "xgb_residual_distribution.png"
    )
    save_feature_importance(
        xgb_best, X_train.columns, "XGBoost", "xgb_feature_importance.png"
    )

save_correlation_heatmap(X_train, "correlation_heatmap.png")
save_model_comparison_bar(metrics_df, "model_comparison")

print("All plots saved successfully.\n")


# =========================================================
# FINAL MODEL SELECTION
# =========================================================
best_model_name = metrics_df.sort_values("R2", ascending=False).iloc[0]["Model"]
print("=" * 60)
print("FINAL BEST MODEL")
print("=" * 60)
print(f"Best model based on highest R2: {best_model_name}")
print()

print("Files saved:")
print(f"- Metrics CSV      : {os.path.join(OUTPUT_DIR, 'model_metrics.csv')}")
print(f"- Summary JSON     : {os.path.join(OUTPUT_DIR, 'summary.json')}")
print(f"- Plots folder     : {PLOT_DIR}")
print(f"- Models folder    : {MODEL_DIR}")
print("=" * 60)