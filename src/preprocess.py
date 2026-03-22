"""
preprocess.py
-------------
Data Cleaning & Preprocessing Pipeline for the Malaysian House Price Dataset.

Issues addressed (identified by inspect_data.py):
    1. Duplicate rows        -- 185 exact duplicates dropped
    2. High-missing columns  -- 10 columns with >75% NaN excluded from features
    3. Price format          -- 'RM 340 000' (space-separated) parsed to float
    4. Property Size format  -- '1000 sq.ft.' suffix stripped, cast to float
    5. Dash as null          -- '-' in Bedroom/Bathroom/Parking Lot/Completion Year
                               replaced with NaN, then cast to numeric
    6. Zero-variance feature -- 'Category' has only 1 unique value, dropped
    7. Outliers              -- Rows outside 1.5xIQR for price & property size removed

Usage:
    from src.preprocess import load_and_preprocess_data
    df = load_and_preprocess_data("data/houses.csv")
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

def clean_price(value):
    """
    Convert Malaysian price string to float.

    The dataset uses space-separated thousands: 'RM 340 000' -> 340000.0
    Note: The original parser only removed commas, which caused ALL rows to
    return NaN because this dataset uses spaces, not commas, as separators.

    Args:
        value: Raw price string or NaN.

    Returns:
        float price in Ringgit Malaysia, or NaN if unparseable.
    """
    if pd.isna(value):
        return np.nan
    # Remove currency prefix and ALL whitespace, then cast
    cleaned = str(value).replace("RM", "").replace(",", "").replace(" ", "").strip()
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_property_size(value):
    """
    Convert property size string to float (sq.ft.).

    Example: '1,200 sq.ft.' -> 1200.0

    Args:
        value: Raw property size string or NaN.

    Returns:
        float area in square feet, or NaN if unparseable.
    """
    if pd.isna(value):
        return np.nan
    cleaned = (
        str(value)
        .lower()
        .replace("sq.ft.", "")
        .replace("sq ft", "")
        .replace(",", "")
        .strip()
    )
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Outlier removal
# ---------------------------------------------------------------------------

def _remove_outliers_iqr(df, column):
    """
    Remove rows where `column` falls outside the 1.5xIQR fence.

    IQR (Interquartile Range) method:
        Lower fence = Q1 - 1.5 * IQR
        Upper fence = Q3 + 1.5 * IQR

    Args:
        df     : DataFrame containing the column.
        column : Name of the numeric column to check.

    Returns:
        DataFrame with outlier rows removed.
    """
    series      = df[column].dropna()
    Q1, Q3      = series.quantile(0.25), series.quantile(0.75)
    IQR         = Q3 - Q1
    lower_fence = Q1 - 1.5 * IQR
    upper_fence = Q3 + 1.5 * IQR
    mask        = df[column].between(lower_fence, upper_fence, inclusive="both")
    return df[mask | df[column].isna()]   # keep rows where column is NaN (handled later)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

# Features selected based on Part 1 design (excluding 'Category' -- zero variance)
SELECTED_FEATURES = [
    "Bedroom",
    "Bathroom",
    "Property Size",
    "Tenure Type",
    "Completion Year",
    "Property Type",
    "Parking Lot",
]

# Numeric columns that use '-' as a null placeholder
DASH_NUMERIC_COLS = ["Bedroom", "Bathroom", "Parking Lot", "Completion Year"]


def load_and_preprocess_data(file_path):
    """
    Load the raw CSV and apply the full cleaning pipeline.

    Steps (in order):
        1.  Load CSV and strip column-name whitespace.
        2.  Drop exact duplicate rows.
        3.  Parse price (target) to float -- fixes space-separator bug.
        4.  Parse Property Size to float  -- strips 'sq.ft.' suffix.
        5.  Replace dash placeholders with NaN in numeric columns.
        6.  Cast numeric columns to float (errors='coerce' for safety).
        7.  Select only the 7 model features + target; drop everything else
            (this implicitly excludes the 10 high-missing columns and
             the zero-variance 'Category' column).
        8.  Drop rows where the target (price) is missing.
        9.  Remove outliers via IQR on price and Property Size.
        10. Reset index.

    Args:
        file_path : Path to the raw houses.csv file.

    Returns:
        Cleaned pandas DataFrame ready for feature selection and model training.
    """

    # -- Step 1: Load & normalise column names --------------------------------
    df = pd.read_csv(file_path)
    df.columns = [col.strip() for col in df.columns]

    # -- Step 2: Drop duplicate rows ------------------------------------------
    before = len(df)
    df = df.drop_duplicates()
    print(f"[Preprocess] Duplicates removed   : {before - len(df)} rows")

    # -- Step 3: Parse price (target variable) --------------------------------
    if "price" not in df.columns:
        raise ValueError("Target column 'price' not found in dataset.")
    df["price"] = df["price"].apply(clean_price)

    # -- Step 4: Parse Property Size ------------------------------------------
    if "Property Size" in df.columns:
        df["Property Size"] = df["Property Size"].apply(clean_property_size)

    # -- Step 5 & 6: Replace '-' placeholders and cast to numeric -------------
    for col in DASH_NUMERIC_COLS:
        if col in df.columns:
            # Replace the dash string with NaN so pd.to_numeric can coerce it
            df[col] = df[col].replace("-", np.nan)
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # -- Step 7: Keep only selected features + target -------------------------
    available_features = [f for f in SELECTED_FEATURES if f in df.columns]
    df = df[available_features + ["price"]]

    # -- Step 8: Drop rows with missing target --------------------------------
    before = len(df)
    df = df.dropna(subset=["price"])
    print(f"[Preprocess] Missing price dropped : {before - len(df)} rows")

    # -- Step 9: Remove outliers (IQR) on price and Property Size -------------
    before = len(df)
    df = _remove_outliers_iqr(df, "price")
    df = _remove_outliers_iqr(df, "Property Size")
    print(f"[Preprocess] Outliers removed      : {before - len(df)} rows")

    # -- Step 10: Reset index -------------------------------------------------
    df = df.reset_index(drop=True)
    print(f"[Preprocess] Final dataset size    : {len(df)} rows x {len(df.columns)} columns")

    return df
