import pandas as pd
import numpy as np


def clean_price(value):
    """Convert price text like 'RM 450,000' to float."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace("RM", "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return np.nan


def clean_property_size(value):
    """Convert property size text like '1,200 sq.ft.' to float."""
    if pd.isna(value):
        return np.nan
    value = str(value).lower().replace("sq.ft.", "").replace("sq ft", "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return np.nan


def load_and_preprocess_data(file_path):
    df = pd.read_csv(file_path)

    # Standardize column names
    df.columns = [col.strip() for col in df.columns]

    # Clean target
    if "price" in df.columns:
        df["price"] = df["price"].apply(clean_price)

    # Clean selected numeric columns if they exist
    if "Property Size" in df.columns:
        df["Property Size"] = df["Property Size"].apply(clean_property_size)

    numeric_cols = ["Bedroom", "Bathroom", "Parking Lot", "Completion Year"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Select a manageable set of features
    selected_features = [
        "Bedroom",
        "Bathroom",
        "Property Size",
        "Category",
        "Tenure Type",
        "Completion Year",
        "Property Type",
        "Parking Lot"
    ]

    selected_features = [col for col in selected_features if col in df.columns]

    # Keep only selected features + target
    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataset.")

    df = df[selected_features + ["price"]]

    # Drop rows with missing target
    df = df.dropna(subset=["price"])

    return df