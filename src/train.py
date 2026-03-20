import os
import joblib
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

from preprocess import load_and_preprocess_data
from evaluate import evaluate_model, plot_actual_vs_predicted


def main():
    # Create output folders if not exist
    os.makedirs("models", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load cleaned data
    df = load_and_preprocess_data("data/houses.csv")

    # Split features and target
    X = df.drop("price", axis=1)
    y = df["price"]

    # Identify column types
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    # Preprocessing
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    # Model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    # Full pipeline
    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train
    pipeline.fit(X_train, y_train)

    # Predict
    y_pred = pipeline.predict(X_test)

    # Evaluate
    evaluate_model(y_test, y_pred, feature_names=X.columns.tolist())

    # Plot actual vs predicted
    plot_actual_vs_predicted(y_test, y_pred)

    # Correlation heatmap for numeric columns only
    numeric_df = df.select_dtypes(include=["int64", "float64"])
    plt.figure(figsize=(8, 6))
    correlation_matrix = numeric_df.corr()
    plt.imshow(correlation_matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(correlation_matrix.columns)), correlation_matrix.columns, rotation=45, ha="right")
    plt.yticks(range(len(correlation_matrix.columns)), correlation_matrix.columns)
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png")
    plt.close()

    # Feature importance plot
    feature_names_after_preprocessing = pipeline.named_steps["preprocessor"].get_feature_names_out()
    importances = pipeline.named_steps["model"].feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names_after_preprocessing,
        "Importance": importances
    }).sort_values(by="Importance", ascending=False).head(10)

    plt.figure(figsize=(10, 6))
    plt.barh(importance_df["Feature"], importance_df["Importance"])
    plt.gca().invert_yaxis()
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.title("Top 10 Feature Importances")
    plt.tight_layout()
    plt.savefig("outputs/feature_importance.png")
    plt.close()

    # Save trained model
    joblib.dump(pipeline, "models/saved_model.pkl")
    print("\nModel saved to models/saved_model.pkl")
    print("Plots saved in outputs/")


if __name__ == "__main__":
    main()