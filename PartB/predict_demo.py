import os
import json
import pickle
import random
import pandas as pd

XGB_AVAILABLE = True
try:
    from xgboost import XGBRegressor
except ImportError:
    XGB_AVAILABLE = False


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

X_test = pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))

with open(os.path.join(OUTPUT_DIR, "summary.json"), "r") as f:
    summary = json.load(f)

rf_r2 = summary["random_forest"]["test_metrics"]["R2"]
xgb_r2 = float("-inf")

if "xgboost" in summary:
    xgb_r2 = summary["xgboost"]["test_metrics"]["R2"]

best_model_name = "random_forest" if rf_r2 >= xgb_r2 else "xgboost"

sample_index = random.randint(0, len(X_test) - 1)
sample_input = X_test.iloc[[sample_index]]

if best_model_name == "random_forest":
    with open(os.path.join(MODEL_DIR, "best_random_forest.pkl"), "rb") as f:
        model = pickle.load(f)
    prediction = model.predict(sample_input)[0]

elif best_model_name == "xgboost" and XGB_AVAILABLE:
    model = XGBRegressor()
    model.load_model(os.path.join(MODEL_DIR, "best_xgboost.json"))
    prediction = model.predict(sample_input)[0]

else:
    raise RuntimeError("Best model could not be loaded.")


def format_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def print_box_line(width=62):
    print("=" * width)


def print_section_title(title, width=62):
    print_box_line(width)
    print(title.center(width))
    print_box_line(width)


def print_features_nicely(row_df):
    row = row_df.iloc[0].to_dict()
    feature_items = list(row.items())

    print("\nINPUT FEATURES")
    print("-" * 62)

    for feature, value in feature_items:
        print(f"{feature:<24} : {format_value(value)}")


print_section_title("PERSON B DEMO SCRIPT")
print(f"Selected Best Model     : {best_model_name}")
print(f"Random Test Row Index   : {sample_index}")
print(f"Predicted House Price   : RM {prediction:,.2f}")

print_features_nicely(sample_input)

print_box_line()