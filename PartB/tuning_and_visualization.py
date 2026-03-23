import json
import os
import pickle
import warnings
from math import sqrt

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold

warnings.filterwarnings("ignore")
plt.ioff()

XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except ImportError:
    XGB_AVAILABLE = False


# =========================================================
# PROJECT PURPOSE
# =========================================================
"""
House Price Prediction - Model Tuning, Evaluation, and Visualization

Main tasks covered by this file:
1. Hyperparameter tuning for the selected regression models.
2. Evaluation using relevant regression metrics.
3. Plot generation for analysis and report inclusion.
4. Saving models, metrics, predictions, and supporting output files.

Why regression metrics are used instead of Precision/Recall/F1:
The target variable is house price, which is a continuous numeric value.
Therefore, this project is a regression task rather than a classification task.
MAE, RMSE, R², and MAPE are more appropriate for measuring regression quality.
"""


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
def load_datasets():
    """Load the preprocessed training and testing datasets."""
    x_train = pd.read_csv(os.path.join(DATA_DIR, "X_train.csv"))
    x_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))
    y_train = pd.read_csv(os.path.join(DATA_DIR, "y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(DATA_DIR, "y_test.csv")).squeeze()
    return x_train, x_test, y_train, y_test


X_train, X_test, y_train, y_test = load_datasets()

print("=" * 60)
print("HOUSE PRICE MODEL TUNING, EVALUATION, AND VISUALIZATION")
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
    """Calculate MAPE while safely ignoring any zero-valued true targets."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mask = y_true != 0
    if mask.sum() == 0:
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_regression(y_true, y_pred, model_name):
    """Compute the main regression metrics used in the project."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error_safe(y_true, y_pred)

    metrics = {
        "Model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape,
    }

    print(f"{model_name} Results")
    print(f"  MAE  : RM {mae:,.2f}")
    print(f"  RMSE : RM {rmse:,.2f}")
    print(f"  R2   : {r2:.4f}")
    print(f"  MAPE : {mape:.2f}%")
    print()

    return metrics


# =========================================================
# PLOT CAPTIONS FOR REPORT USE
# =========================================================
PLOT_CAPTIONS = {
    "rf_actual_vs_predicted_scatter.png": (
        "Random Forest actual vs predicted scatter plot. Points closer to the 45-degree "
        "reference line indicate more accurate predictions."
    ),
    "rf_actual_vs_predicted_line.png": (
        "Random Forest actual and predicted price line plot for the first 100 test samples. "
        "This helps compare trend similarity between true and predicted values."
    ),
    "rf_residual_plot.png": (
        "Random Forest residual plot. Residuals should be distributed around the zero line; "
        "large patterns may indicate bias or underfitting."
    ),
    "rf_residual_distribution.png": (
        "Random Forest residual distribution histogram. A distribution centered near zero "
        "suggests balanced prediction errors."
    ),
    "rf_feature_importance.png": (
        "Top Random Forest feature importance chart. Larger importance scores indicate "
        "features that contribute more strongly to the model's decisions."
    ),
    "xgb_actual_vs_predicted_scatter.png": (
        "XGBoost actual vs predicted scatter plot. Better predictions cluster around the "
        "45-degree reference line."
    ),
    "xgb_actual_vs_predicted_line.png": (
        "XGBoost actual and predicted price line plot for the first 100 test samples. "
        "This visualizes how closely the model follows real price movements."
    ),
    "xgb_residual_plot.png": (
        "XGBoost residual plot. A random spread around zero indicates a healthier error pattern."
    ),
    "xgb_residual_distribution.png": (
        "XGBoost residual distribution histogram. Narrower spread generally indicates more "
        "consistent predictions."
    ),
    "xgb_feature_importance.png": (
        "Top XGBoost feature importance chart. It shows which input variables most influence "
        "predicted house price."
    ),
    "correlation_heatmap.png": (
        "Correlation heatmap of the training features. Darker or more intense values show "
        "stronger linear relationships between variables."
    ),
    "model_comparison_mae.png": (
        "Model comparison chart for MAE. Lower values indicate better average prediction accuracy."
    ),
    "model_comparison_rmse.png": (
        "Model comparison chart for RMSE. Lower values indicate smaller large-error penalties."
    ),
    "model_comparison_r2.png": (
        "Model comparison chart for R². Higher values indicate that the model explains more "
        "variance in house prices."
    ),
    "model_comparison_mape.png": (
        "Model comparison chart for MAPE. Lower percentages indicate smaller average relative error."
    ),
}


# =========================================================
# PLOTTING HELPERS
# =========================================================
def save_actual_vs_predicted_scatter(y_true, y_pred, model_name, filename):
    """Save a scatter plot comparing actual values with predicted values."""
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6, label="Predictions")
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], label="Ideal fit")
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Actual vs Predicted Price — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_actual_vs_predicted_line(y_true, y_pred, model_name, filename, n_points=100):
    """Save a line plot of actual and predicted values for a limited sample window."""
    plot_df = pd.DataFrame({
        "Actual": np.array(y_true),
        "Predicted": np.array(y_pred),
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
    """Save a residual plot to inspect prediction bias and variance patterns."""
    residuals = np.array(y_true) - np.array(y_pred)

    plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, residuals, alpha=0.6, label="Residuals")
    plt.axhline(y=0, label="Zero error")
    plt.xlabel("Predicted Price")
    plt.ylabel("Residuals (Actual - Predicted)")
    plt.title(f"Residual Plot — {model_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_residual_distribution(y_true, y_pred, model_name, filename):
    """Save a histogram of residuals to examine error distribution."""
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
    """Save a horizontal bar chart of the most influential input features."""
    if not hasattr(model, "feature_importances_"):
        return

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False).head(top_n)

    plt.figure(figsize=(10, 6))
    plt.barh(
        importance_df["Feature"][::-1],
        importance_df["Importance"][::-1],
        color="#4c78a8"
    )
    plt.xlabel("Importance Score")
    plt.ylabel("Feature")
    plt.title(f"Top {top_n} Feature Importances — {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_correlation_heatmap(x_data, filename):
    """Save a feature correlation heatmap for descriptive data analysis."""
    corr = x_data.corr(numeric_only=True)

    plt.figure(figsize=(14, 10))
    im = plt.imshow(corr, aspect="auto", cmap="coolwarm")
    plt.colorbar(im)
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOT_DIR, filename), dpi=300)
    plt.close()


def save_model_comparison_bar(metrics_df, filename_prefix):
    """Save comparison bar charts with consistent model colors."""
    metric_names = ["MAE", "RMSE", "R2", "MAPE"]

    color_map = {
        "Random Forest Tuned": "#1f77b4",
        "XGBoost Tuned": "#ff7f0e",
    }

    for metric in metric_names:
        plt.figure(figsize=(8, 6))

        models = metrics_df["Model"]
        values = metrics_df[metric]
        colors = [color_map.get(model, "#888888") for model in models]

        bars = plt.bar(models, values, color=colors)

        for bar in bars:
            height = bar.get_height()
            plt.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom"
            )

        plt.xlabel("Model")
        plt.ylabel(metric)
        plt.title(f"Model Comparison — {metric}")

        handles = [
            plt.Rectangle((0, 0), 1, 1, color=color_map[m])
            for m in color_map if m in models.values
        ]
        labels = [m for m in color_map if m in models.values]

        if handles:
            plt.legend(handles, labels)

        plt.tight_layout()
        plt.savefig(
            os.path.join(PLOT_DIR, f"{filename_prefix}_{metric.lower()}.png"),
            dpi=300
        )
        plt.close()


# =========================================================
# TUNING FUNCTIONS
# =========================================================
def tune_random_forest(x_train, y_train_values):
    """Tune a Random Forest regressor using a compact, reproducible grid search."""
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
        verbose=1,
    )

    grid.fit(x_train, y_train_values)

    print("Best RF Parameters:")
    print(grid.best_params_)
    print(f"Best RF CV Score: {grid.best_score_:.4f}")
    print()

    return grid.best_estimator_, grid.best_params_, grid.best_score_


def tune_xgboost(x_train, y_train_values):
    """Tune an XGBoost regressor using grid search when XGBoost is available."""
    print("=" * 60)
    print("TUNING XGBOOST")
    print("=" * 60)

    xgb = XGBRegressor(
        objective="reg:squarederror",
        random_state=42,
        n_jobs=1,
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
        verbose=1,
    )

    grid.fit(x_train, y_train_values)

    print("Best XGBoost Parameters:")
    print(grid.best_params_)
    print(f"Best XGBoost CV Score: {grid.best_score_:.4f}")
    print()

    return grid.best_estimator_, grid.best_params_, grid.best_score_


# =========================================================
# REPORT SUPPORT HELPERS
# =========================================================
def save_plot_captions_file():
    """Export ready-to-use plot captions for insertion into the written report."""
    captions_path = os.path.join(OUTPUT_DIR, "plot_captions.md")
    with open(captions_path, "w", encoding="utf-8") as file:
        file.write("# Plot Captions\n\n")
        for filename, caption in PLOT_CAPTIONS.items():
            file.write(f"## {filename}\n{caption}\n\n")
    return captions_path


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
metrics_path = os.path.join(OUTPUT_DIR, "model_metrics.csv")
metrics_df.to_csv(metrics_path, index=False)

summary = {
    "problem_type": "regression",
    "metric_note": (
        "This project predicts continuous house prices, so MAE, RMSE, R2, and MAPE "
        "are used instead of Precision, Recall, and F1-score."
    ),
    "random_forest": {
        "best_params": rf_best_params,
        "best_cv_score_r2": rf_best_cv,
        "test_metrics": {
            "MAE": rf_metrics["MAE"],
            "RMSE": rf_metrics["RMSE"],
            "R2": rf_metrics["R2"],
            "MAPE": rf_metrics["MAPE"],
        },
    },
}

if xgb_metrics is not None:
    summary["xgboost"] = {
        "best_params": xgb_params,
        "best_cv_score_r2": xgb_cv,
        "test_metrics": {
            "MAE": xgb_metrics["MAE"],
            "RMSE": xgb_metrics["RMSE"],
            "R2": xgb_metrics["R2"],
            "MAPE": xgb_metrics["MAPE"],
        },
    }

summary_path = os.path.join(OUTPUT_DIR, "summary.json")
with open(summary_path, "w", encoding="utf-8") as file:
    json.dump(summary, file, indent=4)


# =========================================================
# SAVE PREDICTIONS
# =========================================================
rf_pred_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": rf_pred,
    "Residual": np.array(y_test) - np.array(rf_pred),
})
rf_pred_df.to_csv(os.path.join(OUTPUT_DIR, "rf_predictions.csv"), index=False)

if xgb_pred is not None:
    xgb_pred_df = pd.DataFrame({
        "Actual": y_test,
        "Predicted": xgb_pred,
        "Residual": np.array(y_test) - np.array(xgb_pred),
    })
    xgb_pred_df.to_csv(os.path.join(OUTPUT_DIR, "xgb_predictions.csv"), index=False)


# =========================================================
# SAVE MODELS
# =========================================================
with open(os.path.join(MODEL_DIR, "best_random_forest.pkl"), "wb") as file:
    pickle.dump(rf_best, file)

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

captions_path = save_plot_captions_file()
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
print(f"- Metrics CSV   : {metrics_path}")
print(f"- Summary JSON  : {summary_path}")
print(f"- Plots folder  : {PLOT_DIR}")
print(f"- Plot captions : {captions_path}")
print(f"- Models folder : {MODEL_DIR}")
print("=" * 60)