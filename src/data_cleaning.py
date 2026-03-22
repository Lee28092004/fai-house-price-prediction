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

# ── Load raw data ─────────────────────────────────────────────────────────────
print("=" * 60)
print("LOADING DATA")
print("=" * 60)

df = pd.read_csv('data/houses.csv')
print(f"Raw shape: {df.shape[0]} rows × {df.shape[1]} columns")


# ── STEP 1: Drop exact duplicate rows ────────────────────────────────────────
# Problem: 185 rows are exact copies, likely from duplicate scraping.
#          These inflate training data and cause data leakage in train/test splits.
# Fix:     Remove all rows where every column is identical to another row.

print("\n" + "=" * 60)
print("STEP 1 — Removing duplicate rows")
print("=" * 60)

before = len(df)
df = df.drop_duplicates()
removed = before - len(df)
print(f"  Removed {removed} duplicate rows ({removed/before*100:.1f}% of data)")
print(f"  Shape after: {df.shape}")


# ── STEP 2: Replace '-' placeholder with NaN ─────────────────────────────────
# Problem: Many columns use the literal string "-" to represent "no data".
#          Pandas reads this as a valid string, which prevents type casting
#          and breaks isnull() checks.
# Fix:     Replace all "-" with np.nan so they become true missing values.

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


# ── STEP 3: Parse 'price' (target variable) ───────────────────────────────────
# Problem: Target variable is stored as a string: "RM 340 000"
#          Cannot be used in any model or calculation in this format.
# Fix:     Strip "RM" prefix and whitespace, then cast to integer.

print("\n" + "=" * 60)
print("STEP 3 — Parsing 'price' from string to integer")
print("=" * 60)

print(f"  Before — sample values: {df['price'].head(3).tolist()}")
df['price'] = (
    df['price']
    .str.replace('RM', '', regex=False)
    .str.replace(' ', '', regex=False)
    .astype(float)
    .astype('Int64')  # Nullable integer type
)
print(f"  After  — sample values: {df['price'].head(3).tolist()}")
print(f"  Dtype: {df['price'].dtype} | Nulls: {df['price'].isnull().sum()}")
print(f"  Range: RM {df['price'].min():,} – RM {df['price'].max():,}")


# ── STEP 4: Parse 'Property Size' ─────────────────────────────────────────────
# Problem: Size stored as "1000 sq.ft." — the unit suffix prevents numeric use.
# Fix:     Strip " sq.ft." and cast to float.

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
# Problem: Bedroom, Bathroom, Parking Lot, # of Floors, Total Units, and
#          Completion Year all contain numeric data but were loaded as strings
#          (because they also contain '-' placeholders, now converted to NaN).
# Fix:     Use pd.to_numeric() with errors='coerce' — any non-numeric remainder
#          becomes NaN instead of crashing.

print("\n" + "=" * 60)
print("STEP 5 — Casting string-encoded numeric columns to numeric types")
print("=" * 60)

numeric_cols = ['Bedroom', 'Bathroom', 'Parking Lot', '# of Floors', 'Total Units', 'Completion Year']
for col in numeric_cols:
    before_nulls = df[col].isnull().sum()
    df[col] = pd.to_numeric(df[col], errors='coerce')
    after_nulls = df[col].isnull().sum()
    print(f"  {col:<20} → {df[col].dtype}  |  nulls: {after_nulls} ({after_nulls/len(df)*100:.1f}%)")


# ── STEP 6: Drop zero-variance and noise columns ──────────────────────────────
# Problem A: 'Category' has only one unique value across all 4,000 rows
#            ("Apartment / Condominium, For sale") — zero discriminative power.
# Problem B: 'description' is raw scraped marketing text with embedded \r\n,
#            emojis, phone numbers, and URLs. Not a structured feature.
# Fix:       Drop both columns. Description would require a separate NLP pipeline.

print("\n" + "=" * 60)
print("STEP 6 — Dropping zero-variance and noise columns")
print("=" * 60)

cols_to_drop_noise = ['Category', 'description']
df = df.drop(columns=cols_to_drop_noise)
print(f"  Dropped: {cols_to_drop_noise}")
print(f"  Reason — 'Category': 1 unique value, 'description': raw unstructured text")


# ── STEP 7: Drop columns with >80% missing values ─────────────────────────────
# Problem: 8 proximity/location text columns have between 80–97% missing data.
#          Imputing these would be fabricating data, not filling gaps.
# Fix:     Drop any column where >80% of rows are null.

print("\n" + "=" * 60)
print("STEP 7 — Dropping columns with >80% missing values")
print("=" * 60)

missing_pct = df.isnull().mean()
high_missing_cols = missing_pct[missing_pct > 0.80].index.tolist()
print("  Columns dropped (>80% missing):")
for col in high_missing_cols:
    pct = missing_pct[col] * 100
    print(f"    {col:<30} {pct:.1f}% missing")
df = df.drop(columns=high_missing_cols)
print(f"  Shape after: {df.shape}")


# ── STEP 8: Impute remaining missing values ────────────────────────────────────
# Problem: After the above steps, some columns still have missing values.
#          Models cannot train on NaN values.
# Strategy:
#   - Numeric columns  → fill with column median (robust to outliers)
#   - Categorical cols → fill with column mode (most frequent value)

print("\n" + "=" * 60)
print("STEP 8 — Imputing remaining missing values")
print("=" * 60)

numeric_features = df.select_dtypes(include='number').columns.tolist()
# Remove price from imputation — it is the target variable, must not be imputed
if 'price' in numeric_features:
    numeric_features.remove('price')

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
print(f"  Rows removed     : {4000 - df.shape[0]} (duplicates + rows where price was null)")
print(f"  Columns removed  : {32 - df.shape[1]}")

print("\n  Column dtypes after cleaning:")
for col in df.columns:
    print(f"    {col:<25} {str(df[col].dtype):<12} nulls: {df[col].isnull().sum()}")

print("\n  Numeric feature statistics:")
print(df[['Bedroom', 'Bathroom', 'Property Size', 'Parking Lot', 'price']].describe().round(2))


# ── Save clean dataset ────────────────────────────────────────────────────────
output_path = 'data/houses_clean.csv'
df.to_csv(output_path, index=False)
print(f"\n  Clean dataset saved to: {output_path}")
print("  Data cleaning complete. Ready for feature selection.")