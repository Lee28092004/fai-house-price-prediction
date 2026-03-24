"""
Data Cleaning Script — houses.csv
Person A: Data & Model Architect
Step 1 of pipeline: Raw data → Clean dataset

Cleaning steps (in order):
    1. Drop exact duplicate rows
    2. Replace '-' placeholder strings with NaN
    3. Parse 'price' from "RM 340 000" string → integer
    4. Parse 'Property Size' from "1000 sq.ft." string → float
    5. Cast numeric columns stored as strings → numeric dtypes
    6. Drop zero-variance and noise columns
    7. Drop columns with >80% missing values
    8. Impute remaining missing values (median for numeric, mode for categorical)
"""

import pandas as pd
import numpy as np

df = pd.read_csv('data/houses.csv')
original_shape = df.shape
print(f"Raw shape: {original_shape[0]} rows × {original_shape[1]} columns")


# ── STEP 1: Drop exact duplicate rows ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 1 — Removing duplicate rows")
print("=" * 60)

before = len(df)
df = df.drop_duplicates()
removed = before - len(df)
print(f"  Removed {removed} duplicate rows ({removed/before*100:.1f}% of data)")
print(f"  Shape after: {df.shape}")


# ── STEP 2: Replace '-' placeholder with NaN ──────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2 — Replacing '-' placeholder strings with NaN")
print("=" * 60)

dash_counts = {col: (df[col] == '-').sum() for col in df.columns if df[col].dtype == object}
dash_counts = {k: v for k, v in dash_counts.items() if v > 0}
print("  Columns containing '-' placeholders:")
for col, count in sorted(dash_counts.items(), key=lambda x: -x[1]):
    print(f"    {col:<25} {count:>4} occurrences ({count/len(df)*100:.1f}%)")

df = df.replace('-', np.nan)
print(f"  Done. All '-' values replaced with NaN.")


# ── STEP 3: Parse 'price' ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3 — Parsing 'price' from string to integer")
print("=" * 60)

print(f"  Before — sample values: {df['price'].head(3).tolist()}")
df['price'] = (
    df['price']
    .str.replace('RM', '', regex=False)
    .str.replace(' ', '', regex=False)
    .astype(float)
    .astype('Int64')
)
print(f"  After  — sample values: {df['price'].head(3).tolist()}")
print(f"  Dtype: {df['price'].dtype} | Nulls: {df['price'].isnull().sum()}")
print(f"  Range: RM {df['price'].min():,} – RM {df['price'].max():,}")


# ── STEP 4: Parse 'Property Size' ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4 — Parsing 'Property Size' from string to float")
print("=" * 60)

print(f"  Before — sample values: {df['Property Size'].head(3).tolist()}")
df['Property Size'] = (
    df['Property Size']
    .str.replace('sq.ft.', '', regex=False)
    .str.strip()
    .astype(float)
)
print(f"  After  — sample values: {df['Property Size'].head(3).tolist()}")
print(f"  Dtype: {df['Property Size'].dtype} | Nulls: {df['Property Size'].isnull().sum()}")


# ── STEP 5: Cast numeric columns stored as strings ────────────────────────────
print("\n" + "=" * 60)
print("STEP 5 — Casting string-encoded numeric columns to numeric types")
print("=" * 60)

numeric_cols = ['Bedroom', 'Bathroom', 'Parking Lot', '# of Floors', 'Total Units', 'Completion Year']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    after_nulls = df[col].isnull().sum()
    print(f"  {col:<20} → {df[col].dtype}  |  nulls: {after_nulls} ({after_nulls/len(df)*100:.1f}%)")


# ── STEP 6: Drop zero-variance and noise columns ──────────────────────────────
print("\n" + "=" * 60)
print("STEP 6 — Dropping zero-variance and noise columns")
print("=" * 60)

# Category has only 1 unique value; description is raw scraped marketing text
cols_to_drop_noise = ['Category', 'description']
df = df.drop(columns=cols_to_drop_noise)
print(f"  Dropped: {cols_to_drop_noise}")
print(f"  Reason — 'Category': 1 unique value, 'description': raw unstructured text")


# ── STEP 7: Drop columns with >80% missing values ─────────────────────────────
print("\n" + "=" * 60)
print("STEP 7 — Dropping columns with >80% missing values")
print("=" * 60)

missing_pct = df.isnull().mean()
high_missing_cols = missing_pct[missing_pct > 0.80].index.tolist()
print("  Columns dropped (>80% missing):")
for col in high_missing_cols:
    print(f"    {col:<30} {missing_pct[col]*100:.1f}% missing")
df = df.drop(columns=high_missing_cols)
print(f"  Shape after: {df.shape}")


# ── STEP 8: Impute remaining missing values ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8 — Imputing remaining missing values")
print("=" * 60)

numeric_features = df.select_dtypes(include='number').columns.tolist()
# Price is the target variable — must not be imputed
if 'price' in numeric_features:
    numeric_features.remove('price')

try:
    categorical_features = df.select_dtypes(include='str').columns.tolist()
except Exception:
    categorical_features = df.select_dtypes(include='object').columns.tolist()

print("  Numeric columns (median imputation):")
for col in numeric_features:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        median_val = df[col].median()
        df[col] = df[col].fillna(median_val)
        print(f"    {col:<22} → filled {nulls} nulls with median = {median_val}")

print("\n  Categorical columns (mode imputation):")
for col in categorical_features:
    nulls = df[col].isnull().sum()
    if nulls > 0:
        mode_val = df[col].mode()[0]
        df[col] = df[col].fillna(mode_val)
        display_val = str(mode_val)[:40] + "..." if len(str(mode_val)) > 40 else str(mode_val)
        print(f"    {col:<22} → filled {nulls} nulls with mode = '{display_val}'")


# ── Final verification ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("FINAL DATASET SUMMARY")
print("=" * 60)

total_nulls = df.isnull().sum().sum()
print(f"  Final shape      : {df.shape[0]} rows × {df.shape[1]} columns")
print(f"  Total null cells : {total_nulls}")
print(f"  Rows removed     : {original_shape[0] - df.shape[0]} (duplicates + rows where price was null)")
print(f"  Columns removed  : {original_shape[1] - df.shape[1]}")

print("\n  Column dtypes after cleaning:")
for col in df.columns:
    print(f"    {col:<25} {str(df[col].dtype):<12} nulls: {df[col].isnull().sum()}")

print("\n  Numeric feature statistics:")
print(df[['Bedroom', 'Bathroom', 'Property Size', 'Parking Lot', 'price']].describe().round(2))

df.to_csv('data/houses_clean.csv', index=False)
print(f"\n  Clean dataset saved to: data/houses_clean.csv")
print("  Data cleaning complete.")