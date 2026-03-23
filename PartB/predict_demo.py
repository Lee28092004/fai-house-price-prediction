import json
import os
import pickle
import random

import pandas as pd

try:
    from xgboost import XGBRegressor
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


# =========================================================
# PATHS
# =========================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

WIDTH = 74


# =========================================================
# TERMINAL HELPERS
# =========================================================
def clear_screen():
    os.system("cls" if os.name == "nt" else "clear")


def line(char="="):
    print(char * WIDTH)


def box_title(text):
    line("=")
    print(text.center(WIDTH))
    line("=")


def section(text):
    print(f"\n{text}")
    line("-")


def pause():
    input("\nPress Enter to continue...")


def format_value(value):
    if isinstance(value, float):
        return f"{value:,.2f}"
    return str(value)


# =========================================================
# LOADERS
# =========================================================
def load_summary():
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "r", encoding="utf-8") as file:
        return json.load(file)


def load_test_data():
    return pd.read_csv(os.path.join(DATA_DIR, "X_test.csv"))


def load_random_forest_model():
    with open(os.path.join(MODEL_DIR, "best_random_forest.pkl"), "rb") as file:
        return pickle.load(file)


def load_xgboost_model():
    model_path = os.path.join(MODEL_DIR, "best_xgboost.json")

    if not XGB_AVAILABLE or not os.path.exists(model_path):
        return None

    model = XGBRegressor()
    model.load_model(model_path)
    return model


# =========================================================
# DISPLAY
# =========================================================
def print_feature_table(row_df):
    row = row_df.iloc[0].to_dict()

    active_states = []
    normal_features = []

    for feature, value in row.items():
        if feature.startswith("State_"):
            if value == 1 or value is True:
                active_states.append(feature.replace("State_", ""))
        else:
            normal_features.append((feature, value))

    section("INPUT FEATURES")
    for feature, value in normal_features:
        print(f"{feature:<30} : {format_value(value)}")

    if active_states:
        print(f"{'Active State':<30} : {', '.join(active_states)}")


def get_best_model_info(summary):
    rf_r2 = summary["random_forest"]["test_metrics"]["R2"]
    xgb_r2 = summary["xgboost"]["test_metrics"]["R2"] if "xgboost" in summary else float("-inf")

    if rf_r2 >= xgb_r2:
        return "Random Forest", rf_r2, xgb_r2
    return "XGBoost", rf_r2, xgb_r2


def print_prediction_summary(sample_index, rf_pred, xgb_pred, summary):
    best_model, rf_r2, xgb_r2 = get_best_model_info(summary)

    section("PREDICTION RESULTS")
    print(f"{'Selected Row Index':<30} : {sample_index}")
    print(f"{'Random Forest Prediction':<30} : RM {rf_pred:,.2f}")

    if xgb_pred is not None:
        print(f"{'XGBoost Prediction':<30} : RM {xgb_pred:,.2f}")
    else:
        print(f"{'XGBoost Prediction':<30} : Not available")

    section("MODEL COMPARISON")
    print(f"{'Best Model (by R²)':<30} : {best_model}")
    print(f"{'Random Forest R²':<30} : {rf_r2:.4f}")

    if "xgboost" in summary:
        print(f"{'XGBoost R²':<30} : {xgb_r2:.4f}")


def show_project_notes():
    clear_screen()
    box_title("PROJECT OVERVIEW")

    print("This system predicts house prices using machine learning models.")
    print()
    print("Workflow of the system:")
    print("- Load cleaned and preprocessed dataset")
    print("- Train and tune multiple regression models")
    print("- Evaluate model performance using MAE, RMSE, R², and MAPE")
    print("- Select the best model based on performance")
    print("- Generate predictions for unseen data")
    print()
    print("Demonstration purpose:")
    print("- Select a sample row from the dataset")
    print("- Display its input features")
    print("- Show predicted house price")
    print("- Compare model performance")
    print()
    print("Evaluation note:")
    print("- This is a regression task (continuous output)")
    print("- Regression metrics are used instead of classification metrics")
    
    pause()

# =========================================================
# INPUT
# =========================================================
def choose_mode():
    while True:
        clear_screen()
        box_title("HOUSE PRICE PREDICTION DEMO")
        print("1. Predict using a random test row")
        print("2. Predict using a manual row index")
        print("3. Show project notes")
        print("4. Exit")

        choice = input("\nEnter choice [1-4]: ").strip()

        if choice in {"1", "2", "3", "4"}:
            return choice

        print("\nInvalid choice. Please select 1, 2, 3, or 4.")
        pause()


def get_manual_index(max_index):
    while True:
        user_input = input(f"Enter row index (0 to {max_index}): ").strip()
        try:
            index = int(user_input)
            if 0 <= index <= max_index:
                return index
            print("Index out of range.")
        except ValueError:
            print("Please enter a valid whole number.")


# =========================================================
# PREDICTION FLOW
# =========================================================
def run_prediction(x_test, summary, rf_model, xgb_model, mode):
    if mode == "1":
        sample_index = random.randint(0, len(x_test) - 1)
    else:
        sample_index = get_manual_index(len(x_test) - 1)

    sample_input = x_test.iloc[[sample_index]]

    rf_prediction = rf_model.predict(sample_input)[0]
    xgb_prediction = xgb_model.predict(sample_input)[0] if xgb_model is not None else None

    clear_screen()
    box_title("HOUSE PRICE PREDICTION RESULT")
    print_prediction_summary(sample_index, rf_prediction, xgb_prediction, summary)
    print_feature_table(sample_input)
    pause()


# =========================================================
# MAIN
# =========================================================
def main():
    try:
        x_test = load_test_data()
        summary = load_summary()
        rf_model = load_random_forest_model()
        xgb_model = load_xgboost_model()
    except FileNotFoundError as error:
        clear_screen()
        box_title("ERROR")
        print("A required file is missing.")
        print(f"Missing file: {error.filename}")
        print("\nPlease run tuning_and_visualization.py first.")
        return
    except Exception as error:
        clear_screen()
        box_title("ERROR")
        print(f"Unexpected error: {error}")
        return

    while True:
        choice = choose_mode()

        if choice == "4":
            clear_screen()
            box_title("EXIT")
            print("Goodbye.")
            break

        if choice == "3":
            show_project_notes()
            continue

        run_prediction(x_test, summary, rf_model, xgb_model, choice)


if __name__ == "__main__":
    main()