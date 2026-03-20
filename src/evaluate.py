import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def adjusted_r2_score(r2, n, p):
    """Calculate Adjusted R²."""
    if n <= p + 1:
        return None
    return 1 - (1 - r2) * (n - 1) / (n - p - 1)


def evaluate_model(y_true, y_pred, feature_names=None):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    p = len(feature_names) if feature_names is not None else 1
    n = len(y_true)
    adj_r2 = adjusted_r2_score(r2, n, p)

    print("Model Evaluation Results")
    print("------------------------")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²: {r2:.4f}")
    if adj_r2 is not None:
        print(f"Adjusted R²: {adj_r2:.4f}")

    return mae, rmse, r2, adj_r2


def plot_actual_vs_predicted(y_true, y_pred, output_path="outputs/actual_vs_predicted.png"):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_true, y_pred, alpha=0.6)
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title("Actual vs Predicted House Prices")

    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")

    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()