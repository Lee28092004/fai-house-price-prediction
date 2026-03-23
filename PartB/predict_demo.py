import os
import json
import pickle
import random
import pandas as pd
from xgboost import XGBRegressor

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

# Load Random Forest
with open(os.path.join(MODEL_DIR, "best_random_forest.pkl"), "rb") as f:
    rf_model = pickle.load(f)

rf_prediction = rf_model.predict(sample_input)[0]

# Load XGBoost if available in summary
xgb_prediction = None
if "xgboost" in summary:
    xgb_model = XGBRegressor()
    xgb_model.load_model(os.path.join(MODEL_DIR, "best_xgboost.json"))
    xgb_prediction = xgb_model.predict(sample_input)[0]


def format_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


def print_box_line(width=64):
    print("=" * width)


def print_sub_line(width=64):
    print("-" * width)


def print_section_title(title, width=64):
    print_box_line(width)
    print(title.center(width))
    print_box_line(width)


def print_features_nicely(row_df):
    row = row_df.iloc[0].to_dict()

    print("\nINPUT FEATURES")
    print_sub_line()

    active_states = []
    normal_features = []

    for feature, value in row.items():
        if feature.startswith("State_"):
            if value is True or value == 1:
                active_states.append(feature.replace("State_", ""))
        else:
            normal_features.append((feature, value))

    for feature, value in normal_features:
        print(f"{feature:<26} : {format_value(value)}")

    if active_states:
        print(f"{'Active State':<26} : {', '.join(active_states)}")


print_section_title("PERSON B DEMO SCRIPT")
print(f"Random Test Row Index      : {sample_index}")

print("\nMODEL PREDICTIONS")
print_sub_line()
print(f"Random Forest Prediction   : RM {rf_prediction:,.2f}")

if xgb_prediction is not None:
    print(f"XGBoost Prediction         : RM {xgb_prediction:,.2f}")

print("\nBEST MODEL SELECTED")
print_sub_line()
print(f"{best_model_name}")

print_features_nicely(sample_input)

print_box_line()