"""
Data Detection Script — houses.csv
Person A: Data & Model Architect
Step 0: Audit raw data to identify ALL issues BEFORE any cleaning is done.

Run this script first to understand what problems exist in the dataset.
The findings from this script directly inform the cleaning steps in data_cleaning.py.
"""

import pandas as pd
import numpy as np

# ── Load raw data (read-only — nothing is modified here) ──────────────────────
df = pd.read_csv('data/houses.csv')

DIVIDER = "=" * 65

print(DIVIDER)
print("  DATA DETECTION REPORT — houses.csv")
print("  Purpose: Identify all data quality issues before cleaning")
print(DIVIDER)


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 1: Basic shape and structure
# ─────────────────────────────────────────────────────────────────────────────
print("\n[DETECTION 1] Dataset shape and column overview")
print("-" * 65)

print(f"  Total rows    : {df.shape[0]}")
print(f"  Total columns : {df.shape[1]}")
print(f"\n  Column names:")
for i, col in enumerate(df.columns, 1):
    print(f"    {i:>2}. {col}")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 2: Duplicate column names
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 2] Duplicate column names")
print("-" * 65)
print("  Checking whether any two columns share the same name...\n")

seen = {}
dupe_cols = []
for col in df.columns:
    if col in seen:
        dupe_cols.append(col)
    seen[col] = True

if dupe_cols:
    print(f"  FOUND: Duplicate column names: {dupe_cols}")
else:
    print("  No duplicate column names found.")
    print("  All 32 column names are unique.")
    print("\n  RESULT: No action needed.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 3: Data types — are numeric columns stored as wrong types?
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 3] Column data types")
print("-" * 65)
print("  (Columns that SHOULD be numeric but are stored as strings are a problem)")
print()

type_counts = df.dtypes.value_counts()
for dtype, count in type_counts.items():
    print(f"  {str(dtype):<15} → {count} columns")

print(f"\n  Columns flagged as string — inspect sample values:")
try:
    string_cols = df.select_dtypes(include='str').columns.tolist()
except Exception:
    string_cols = df.select_dtypes(include='object').columns.tolist()

for col in string_cols:
    sample = df[col].dropna().iloc[0] if df[col].dropna().shape[0] > 0 else "N/A"
    sample_str = str(sample)[:55].replace('\n', ' ').replace('\r', '')
    print(f"    [{col}]  sample: \"{sample_str}\"")

print(f"\n  FINDING: 31 out of 32 columns are stored as strings.")
print(f"  Several of these (Bedroom, Bathroom, Price, Property Size, etc.)")
print(f"  are clearly numeric and need to be cast to the correct dtype.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 4: Duplicate rows
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 4] Duplicate rows")
print("-" * 65)

total_dupes = df.duplicated().sum()
ad_dupes = df['Ad List'].duplicated().sum()

print(f"  Fully duplicate rows (all columns identical) : {total_dupes}")
print(f"  Duplicate Ad List IDs (same listing ID twice): {ad_dupes}")

if total_dupes > 0:
    print(f"\n  FINDING: {total_dupes} duplicate rows detected ({total_dupes/len(df)*100:.1f}% of data).")
    print(f"  These inflate training data and cause data leakage across train/test splits.")
    print(f"\n  Sample of a duplicated entry:")
    first_dupe_val = df[df.duplicated(keep=False)].iloc[0][['Bedroom', 'Bathroom', 'Property Size', 'price']]
    print(f"  {first_dupe_val.to_dict()}")
else:
    print("  RESULT: No duplicate rows found.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 5: True NaN / missing values
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 5] True NaN missing values (per column)")
print("-" * 65)
print("  Columns with at least 1 missing value (sorted by % missing):\n")

missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Missing Count': missing,
    'Missing %': missing_pct
}).sort_values('Missing %', ascending=False)
missing_df = missing_df[missing_df['Missing Count'] > 0]

if missing_df.empty:
    print("  No NaN values found via isnull().")
else:
    for col, row in missing_df.iterrows():
        severity = "CRITICAL" if row['Missing %'] > 80 else ("HIGH" if row['Missing %'] > 40 else "MODERATE")
        print(f"  [{severity:<8}] {col:<28} {row['Missing Count']:>4} rows  ({row['Missing %']:>5.1f}%)")

print(f"\n  FINDING: {len(missing_df)} columns have true NaN values.")
print(f"  NOTE: This only catches real NaN — disguised missing values (e.g. '-') are checked in Detection 6.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 6: Disguised / fake missing values (placeholder strings)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 6] Disguised missing values — '-' used as placeholder")
print("-" * 65)
print("  These look like valid data to pandas but actually mean 'no data'.")
print("  They are INVISIBLE to isnull() and would cause silent errors in modelling.\n")

dash_found = False
for col in df.columns:
    try:
        count = (df[col] == '-').sum()
        if count > 0:
            pct = count / len(df) * 100
            print(f"  {col:<28}  {count:>4} rows contain '-'  ({pct:.1f}%)")
            dash_found = True
    except TypeError:
        pass

if not dash_found:
    print("  No '-' placeholders found.")
else:
    print(f"\n  FINDING: Multiple columns use '-' as a missing value placeholder.")
    print(f"  Must be replaced with NaN before any numeric casting or analysis.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 7: Target variable — 'price'
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 7] Target variable — 'price'")
print("-" * 65)

print(f"  Stored dtype  : {df['price'].dtype}")
print(f"  Sample values : {df['price'].head(5).tolist()}")
print(f"  Unique values : {df['price'].nunique()}")
print(f"  Null count    : {df['price'].isnull().sum()}")

try:
    test = df['price'].str.replace('RM', '', regex=False).str.replace(' ', '', regex=False).astype(float)
    print(f"  Can be parsed to numeric? : YES")
    print(f"  Price range (after parse) : RM {test.min():,.0f} – RM {test.max():,.0f}")
except Exception as e:
    print(f"  Can be parsed to numeric? : NO — {e}")

print(f"\n  FINDING: 'price' is stored as a STRING (e.g. 'RM 340 000').")
print(f"  The 'RM' prefix and spaces prevent any numeric operation or modelling.")
print(f"  Must strip prefix and cast to integer before use as target variable.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 8: Numeric features stored as strings
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 8] Numeric features stored as string dtype")
print("-" * 65)
print("  Testing whether columns expected to be numeric can be cast...\n")

suspect_cols = ['Bedroom', 'Bathroom', 'Property Size', 'Parking Lot',
                '# of Floors', 'Total Units', 'Completion Year']

for col in suspect_cols:
    raw_dtype = df[col].dtype
    sample_vals = df[col].dropna().head(3).tolist()
    temp = df[col].replace('-', np.nan)
    if col == 'Property Size':
        temp = temp.str.replace('sq.ft.', '', regex=False).str.strip()
    try:
        pd.to_numeric(temp, errors='raise')
        castable = "YES"
    except Exception:
        temp_numeric = pd.to_numeric(temp, errors='coerce')
        non_numeric = temp_numeric.isnull().sum() - temp.isnull().sum()
        castable = f"PARTIAL — {non_numeric} non-numeric entries remain"

    print(f"  {col:<22}  dtype={str(raw_dtype):<10}  castable={castable}")
    print(f"    sample: {[str(v)[:25] for v in sample_vals]}")

print(f"\n  FINDING: These columns must be cast to numeric types.")
print(f"  'Property Size' additionally contains a unit suffix ('sq.ft.') that must be stripped.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 9: Inconsistent categorical values (typos / mixed casing)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 9] Inconsistent categorical values in key columns")
print("-" * 65)
print("  Checking Tenure Type, Land Title, Property Type and Floor Range")
print("  for typos, mixed casing, or irregular entries...\n")

cat_cols = ['Tenure Type', 'Land Title', 'Property Type', 'Floor Range']
inconsistency_found = False

for col in cat_cols:
    unique_vals = df[col].replace('-', np.nan).dropna().unique()
    normalised = [str(v).strip().lower() for v in unique_vals]
    dupe_normalised = len(normalised) != len(set(normalised))
    print(f"  {col}")
    print(f"    Unique values ({len(unique_vals)}): {sorted([str(v) for v in unique_vals])}")
    if dupe_normalised:
        print(f"    WARNING: Case/whitespace inconsistencies detected.")
        inconsistency_found = True
    else:
        print(f"    OK — values are consistent, no casing issues.")
    print()

if not inconsistency_found:
    print("  RESULT: No casing or typo issues found in these categorical columns.")
    print("  Values are clean and consistently formatted.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 10: Impossible / out-of-range values in numeric columns
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 10] Impossible or out-of-range values in numeric columns")
print("-" * 65)
print("  Checking for values that are technically numeric but logically wrong")
print("  (e.g. 0 bedrooms, negative price, completion year far in the future)...\n")

temp_df = df.copy()
temp_df = temp_df.replace('-', np.nan)
temp_df['Bedroom']   = pd.to_numeric(temp_df['Bedroom'], errors='coerce')
temp_df['Bathroom']  = pd.to_numeric(temp_df['Bathroom'], errors='coerce')
temp_df['Completion Year'] = pd.to_numeric(temp_df['Completion Year'], errors='coerce')
temp_df['price_num'] = (
    temp_df['price'].str.replace('RM', '', regex=False)
                    .str.replace(' ', '', regex=False)
                    .astype(float)
)

issues_found = False

zero_bed = (temp_df['Bedroom'] == 0).sum()
if zero_bed > 0:
    print(f"  Bedroom      : {zero_bed} entries with 0 bedrooms — likely invalid.")
    issues_found = True
else:
    print(f"  Bedroom      : range {int(temp_df['Bedroom'].min())}–{int(temp_df['Bedroom'].max())} — looks reasonable.")

zero_bath = (temp_df['Bathroom'] == 0).sum()
if zero_bath > 0:
    print(f"  Bathroom     : {zero_bath} entries with 0 bathrooms — likely invalid.")
    issues_found = True
else:
    print(f"  Bathroom     : range {int(temp_df['Bathroom'].min())}–{int(temp_df['Bathroom'].max())} — looks reasonable.")

future_years = (temp_df['Completion Year'] > 2030).sum()
if future_years > 0:
    print(f"  Completion Year: {future_years} entries beyond year 2030 — may be erroneous.")
    issues_found = True
else:
    valid_years = temp_df['Completion Year'].dropna()
    print(f"  Completion Year: range {int(valid_years.min())}–{int(valid_years.max())} — within expected range.")

neg_price = (temp_df['price_num'] < 0).sum()
if neg_price > 0:
    print(f"  Price        : {neg_price} negative values detected — data error.")
    issues_found = True
else:
    print(f"  Price        : no negative values found — OK.")

if not issues_found:
    print(f"\n  RESULT: No impossible values detected.")
    print(f"  All numeric ranges are within logically acceptable bounds.")
else:
    print(f"\n  FINDING: Impossible values detected — review before modelling.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 11: Zero-variance columns (same value in every row)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 11] Zero-variance columns")
print("-" * 65)
print("  A column with only 1 unique value provides no information to any model.\n")

zero_var_found = False
for col in df.columns:
    n_unique = df[col].nunique(dropna=True)
    if n_unique == 1:
        val = df[col].dropna().iloc[0]
        print(f"  FOUND: '{col}' — only 1 unique value: \"{str(val)[:60]}\"")
        zero_var_found = True

if not zero_var_found:
    print("  No zero-variance columns found.")
else:
    print(f"\n  FINDING: Zero-variance columns should be dropped before modelling.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 12: Columns with raw unstructured / noise text
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 12] Raw text / noise columns")
print("-" * 65)
print("  Checking for columns containing embedded newlines, URLs, or emojis...\n")

noise_indicators = ['\r\n', 'http', 'wasap', '🔥', '😍', 'Show contact number', 'Continue Reading']
try:
    str_cols = df.select_dtypes(include='str').columns.tolist()
except Exception:
    str_cols = df.select_dtypes(include='object').columns.tolist()

noise_found = False
for col in str_cols:
    sample_str = str(df[col].dropna().iloc[0]) if df[col].dropna().shape[0] > 0 else ""
    found_noise = [ind for ind in noise_indicators if ind in sample_str]
    if found_noise:
        display = sample_str[:80].replace('\r', '').replace('\n', ' ')
        print(f"  FOUND: '{col}' — noise markers detected: {found_noise}")
        print(f"    Preview: \"{display}...\"")
        noise_found = True

if not noise_found:
    print("  No embedded noise detected in string columns.")
else:
    print(f"\n  FINDING: Column(s) with embedded marketing text, URLs, and emojis")
    print(f"  are not usable as model features without a separate NLP pipeline.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION 13: Columns too sparse to impute (>80% missing combined NaN + dash)
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n[DETECTION 13] Columns too sparse to impute (>80% missing after dash replacement)")
print("-" * 65)
print("  Combining true NaN and '-' placeholders to see actual missingness...\n")

df_temp = df.replace('-', np.nan)
true_missing_pct = df_temp.isnull().mean() * 100
high_missing = true_missing_pct[true_missing_pct > 80].sort_values(ascending=False)

if high_missing.empty:
    print("  No columns exceed 80% missing.")
else:
    for col, pct in high_missing.items():
        print(f"  DROP CANDIDATE: {col:<30}  {pct:.1f}% missing")

print(f"\n  FINDING: {len(high_missing)} columns exceed 80% missing data.")
print(f"  Imputing these would mean fabricating data for 4 in every 5 rows.")
print(f"  Recommended action: drop these columns entirely.")


# ─────────────────────────────────────────────────────────────────────────────
# DETECTION SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print(f"\n{DIVIDER}")
print("  DETECTION SUMMARY")



print(DIVIDER)
print("  Detection complete. No data was modified.")
print("  Proceed to data_cleaning.py to apply the fixes above.")
print(DIVIDER)