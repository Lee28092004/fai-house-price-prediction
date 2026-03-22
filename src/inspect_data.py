"""
inspect_data.py
---------------
Data Quality Inspection Script for the Malaysian House Price Dataset.

Purpose:
    Before any cleaning is performed, this script audits the raw dataset and
    produces a structured report of every data-quality issue found.  The
    findings here directly drive the cleaning decisions made in preprocess.py.

Run:
    py src/inspect_data.py
"""

import pandas as pd
import numpy as np

# ── 1. Load raw data ──────────────────────────────────────────────────────────
df = pd.read_csv("data/houses.csv")
df.columns = [col.strip() for col in df.columns]   # strip any whitespace from headers

TOTAL_ROWS = len(df)
TOTAL_COLS = len(df.columns)

print("=" * 65)
print("  HOUSE PRICE DATASET - DATA QUALITY INSPECTION REPORT")
print("=" * 65)
print(f"\nDataset shape : {TOTAL_ROWS} rows x {TOTAL_COLS} columns")
print(f"Columns       : {df.columns.tolist()}\n")


# ── 2. Duplicate Rows ─────────────────────────────────────────────────────────
print("-" * 65)
print("ISSUE 1 -- DUPLICATE ROWS")
print("-" * 65)
n_dupes = df.duplicated().sum()
print(f"  Exact duplicate rows found : {n_dupes} ({n_dupes / TOTAL_ROWS * 100:.1f}%)")
print("  Action: Drop all duplicates, keep first occurrence.\n")


# ── 3. Missing Values ─────────────────────────────────────────────────────────
print("-" * 65)
print("ISSUE 2 -- MISSING VALUES (NaN)")
print("-" * 65)
missing = df.isnull().sum()
pct     = (missing / TOTAL_ROWS * 100).round(2)
missing_report = (
    pd.DataFrame({"Missing Count": missing, "Missing %": pct})
    .query("`Missing Count` > 0")
    .sort_values("Missing %", ascending=False)
)
print(missing_report.to_string())
print("""
  Observation:
    - 10 columns exceed 75% missing (Highway 96% -> School 76%).
    - These columns carry almost no usable signal and will be DROPPED.
    - Firm Type / Firm Number / REN Number (~5% missing) are not used
      as model features, so they are also excluded.\n""")


# ── 4. Price Column — Format & Parser Bug ────────────────────────────────────
print("-" * 65)
print("ISSUE 3 -- PRICE COLUMN: SPACE-SEPARATED FORMAT + BROKEN PARSER")
print("-" * 65)
sample_prices = df["price"].dropna().head(8).tolist()
print(f"  Raw samples : {sample_prices}")

# Demonstrate that the ORIGINAL parser (comma-strip only) fails on this data
def _old_clean_price(value):
    """Original parser -- strips commas only."""
    if pd.isna(value):
        return np.nan
    value = str(value).replace("RM", "").replace(",", "").strip()
    try:
        return float(value)
    except ValueError:
        return np.nan

old_results = df["price"].apply(_old_clean_price)
n_failed    = old_results.isna().sum()
print(f"\n  Original parser result  : {n_failed} / {TOTAL_ROWS} rows return NaN")
print("  Root cause: prices use SPACES as thousand separators ('RM 340 000'),")
print("              NOT commas ('RM 340,000').  float('340 000') raises ValueError.")
print("  Action: Replace 'RM' AND all spaces, then cast to float.\n")


# ── 5. Property Size — String Format ─────────────────────────────────────────
print("-" * 65)
print("ISSUE 4 -- PROPERTY SIZE: STRING FORMAT")
print("-" * 65)
size_samples = df["Property Size"].dropna().unique()[:8].tolist()
print(f"  Raw samples : {size_samples}")
print("  Format      : '<number> sq.ft.' -- suffix must be stripped before numeric cast.\n")


# ── 6. Dash ('-') Used as Null Placeholder ───────────────────────────────────
print("-" * 65)
print("ISSUE 5 -- DASH ('-') USED AS NULL PLACEHOLDER")
print("-" * 65)
dash_cols = ["Bedroom", "Bathroom", "Parking Lot", "Completion Year"]
for col in dash_cols:
    n_dash = (df[col].astype(str).str.strip() == "-").sum()
    pct_d  = n_dash / TOTAL_ROWS * 100
    print(f"  {col:<20}: {n_dash} dashes  ({pct_d:.1f}%)")
print("""
  Observation:
    - Parking Lot has the most dashes (~30%) -- likely means 'none provided'.
    - Completion Year dashes (~50%) mean the year is unknown/unrecorded.
  Action: Replace '-' with NaN, then coerce column to numeric type.\n""")


# ── 7. Zero-Variance Feature ─────────────────────────────────────────────────
print("-" * 65)
print("ISSUE 6 -- ZERO-VARIANCE (CONSTANT) FEATURE")
print("-" * 65)
cat_unique = df["Category"].unique()
print(f"  'Category' unique values : {cat_unique}")
print("  All 4,000 rows share the same value -> zero predictive power.")
print("  Action: Drop 'Category' from the feature set.\n")


# ── 8. Outliers ───────────────────────────────────────────────────────────────
print("-" * 65)
print("ISSUE 7 -- OUTLIERS (IQR METHOD)")
print("-" * 65)

def _parse_price(v):
    if pd.isna(v): return np.nan
    v = str(v).replace("RM", "").replace(" ", "").replace(",", "").strip()
    try: return float(v)
    except: return np.nan

def _parse_size(v):
    if pd.isna(v): return np.nan
    v = str(v).lower().replace("sq.ft.", "").replace("sq ft", "").replace(",", "").strip()
    try: return float(v)
    except: return np.nan

price_num = df["price"].apply(_parse_price)
size_num  = df["Property Size"].apply(_parse_size)

for label, series in [("price (RM)", price_num), ("Property Size (sq.ft.)", size_num)]:
    Q1, Q3  = series.quantile(0.25), series.quantile(0.75)
    IQR     = Q3 - Q1
    lo, hi  = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
    n_out   = int(((series < lo) | (series > hi)).sum())
    top5    = sorted(series.dropna().tolist(), reverse=True)[:5]
    print(f"  {label}")
    print(f"    IQR bounds : [{lo:,.0f}  --  {hi:,.0f}]")
    print(f"    Outliers   : {n_out} rows ({n_out / TOTAL_ROWS * 100:.1f}%)")
    print(f"    Top 5 high : {[f'{v:,.0f}' for v in top5]}")

# Extra: suspicious property sizes < 50 sq.ft.
tiny = int((size_num < 50).sum())
print(f"\n  Property sizes < 50 sq.ft. (likely data-entry errors) : {tiny} rows")
print("  Action: Remove rows where price OR property size fall outside 1.5xIQR.\n")


# ── 9. Summary Table ──────────────────────────────────────────────────────────
print("=" * 65)
print("  CLEANING ACTION PLAN SUMMARY")
print("=" * 65)
actions = [
    ("1", "Duplicate rows",           f"{n_dupes} rows",        "Drop duplicates"),
    ("2", "High-missing columns",     "10 columns >75% NaN",    "Exclude from features"),
    ("3", "Price format (RM X X X)",  "All 4000 rows",          "Strip 'RM' + spaces, cast float"),
    ("4", "Property Size format",     "All 4000 rows",          "Strip 'sq.ft.', cast float"),
    ("5", "Dash as null placeholder", "4 numeric columns",      "Replace '-' -> NaN, cast numeric"),
    ("6", "Zero-variance Category",   "1 column",               "Drop column"),
    ("7", "Outliers (IQR)",           "Price + Property Size",  "Remove rows outside 1.5xIQR"),
]
print(f"\n  {'#':<3} {'Issue':<32} {'Scope':<26} {'Action'}")
print(f"  {'---':<3} {'---':<32} {'---':<26} {'---'}")
for num, issue, scope, action in actions:
    print(f"  {num:<3} {issue:<32} {scope:<26} {action}")
print()
