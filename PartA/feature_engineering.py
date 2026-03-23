"""
Feature Engineering Script — houses_clean.csv
Person A: Data & Model Architect
Step 3 of pipeline: Selected features → encoded + engineered dataset → Train/Test split

Reads feature_selection_result.json then:
    1. Caps Property Size outlier at 99th percentile
    2. Engineers Property Age from Completion Year
    3. Extracts State from Address string
    4. Engineers Transit Zone binary flag from State
    5. Engineers Facility Count and Facility Tier from Facilities string
    6. Encodes categoricals (label + one-hot)
    7. Assembles final feature matrix
    8. Splits 80/20 Train/Test
    9. Saves all outputs
"""

import json
import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Load inputs ────────────────────────────────────────────────────────────────
df = pd.read_csv('data/houses_clean.csv')
with open('data/feature_selection_result.json') as f:
    selection = json.load(f)

print(f"Input: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Selected from feature_selection.py: {selection['selected_numeric'] + selection['selected_categorical']}\n")


# ── Step 1: Outlier cap on Property Size ──────────────────────────────────────
original_max = df['Property Size'].max()
cap = df['Property Size'].quantile(0.99)
df['Property Size'] = df['Property Size'].clip(upper=cap)
print(f"Property Size capped at 99th percentile: {cap:,.0f} sq.ft.  (original max was {original_max:,.0f})")


# ── Step 2: Property Age ──────────────────────────────────────────────────────
current_year = datetime.date.today().year
df['Property Age'] = (current_year - df['Completion Year']).clip(lower=0)
print(f"Property Age engineered: {current_year} - Completion Year, range {int(df['Property Age'].min())}–{int(df['Property Age'].max())} years")


# ── Step 3: State from Address ────────────────────────────────────────────────
STATE_MAP = {
    'Kajang': 'Selangor', 'Seri Kembangan': 'Selangor', 'Klang': 'Selangor',
    'Ampang': 'Selangor', 'Cyberjaya': 'Selangor', 'Cheras': 'Selangor',
    'Puchong': 'Selangor', 'Subang Jaya': 'Selangor', 'Petaling Jaya': 'Selangor',
    'Shah Alam': 'Selangor', 'Bangi': 'Selangor', 'Sepang': 'Selangor',
    'Bukit Jalil': 'Kuala Lumpur', 'Sentul': 'Kuala Lumpur',
    'Sungai Besi': 'Kuala Lumpur', 'Kepong': 'Kuala Lumpur',
    'Johor Bahru': 'Johor', 'Skudai': 'Johor',
    'Kota Kinabalu': 'Sabah', 'Kuching': 'Sarawak',
}
VALID_STATES = {
    'Kuala Lumpur', 'Selangor', 'Penang', 'Johor', 'Sabah', 'Sarawak',
    'Putrajaya', 'Melaka', 'Negeri Sembilan', 'Pahang', 'Perak',
    'Kedah', 'Kelantan', 'Terengganu', 'Perlis',
}

def extract_state(addr):
    last = [p.strip() for p in str(addr).split(',')][-1]
    mapped = STATE_MAP.get(last, last)
    return mapped if mapped in VALID_STATES else 'Other'

df['State'] = df['Address'].apply(extract_state)
print(f"State extracted: {df['State'].nunique()} unique states")


# ── Step 4: Transit Zone ──────────────────────────────────────────────────────
TRANSIT_STATES = {'Kuala Lumpur', 'Selangor', 'Putrajaya'}
df['Transit Zone'] = df['State'].apply(lambda s: 1 if s in TRANSIT_STATES else 0)
print(f"Transit Zone: {df['Transit Zone'].sum()} in Klang Valley, {(df['Transit Zone']==0).sum()} outside")


# ── Step 5: Facility Count + Tier ────────────────────────────────────────────
def count_fac(s):
    return len([x.strip() for x in str(s).split(',') if x.strip()]) if pd.notna(s) else 0

def tier(n):
    return 0 if n <= 3 else (1 if n <= 7 else 2)

df['Facility Count'] = df['Facilities'].apply(count_fac)
df['Facility Tier']  = df['Facility Count'].apply(tier)
print(f"Facility Count: range {int(df['Facility Count'].min())}–{int(df['Facility Count'].max())}, mean {df['Facility Count'].mean():.1f}")
print(f"Facility Tier : Basic={( df['Facility Tier']==0).sum()}, Standard={(df['Facility Tier']==1).sum()}, Premium={(df['Facility Tier']==2).sum()}\n")


# ── Step 6: Encode categoricals ───────────────────────────────────────────────
# Tenure Type: binary (Freehold=1, Leasehold=0)
df['Tenure Type Encoded'] = df['Tenure Type'].map({'Freehold': 1, 'Leasehold': 0})

# Land Title: ordinal (Non-Bumi=2, Bumi=1, Malay Reserved=0)
df['Land Title Encoded'] = df['Land Title'].map({'Non Bumi Lot': 2, 'Bumi Lot': 1, 'Malay Reserved': 0})

# Property Type: label encode
le = LabelEncoder()
df['Property Type Encoded'] = le.fit_transform(df['Property Type'])
print(f"Property Type classes: {dict(zip(le.classes_, range(len(le.classes_))))}")

# State: one-hot (drop_first removes Johor as baseline)
state_dummies = pd.get_dummies(df['State'], prefix='State', drop_first=True)
df = pd.concat([df, state_dummies], axis=1)
print(f"State dummies: {list(state_dummies.columns)}\n")


# ── Step 7: Assemble feature matrix ───────────────────────────────────────────
# Selected numeric and categorical columns come directly from feature_selection_result.json.
# Engineered features (Property Age, Transit Zone, Facility Count, Facility Tier)
# and encoded columns replace their raw source columns in the final matrix.
selected_numeric     = selection['selected_numeric']
selected_categorical = selection['selected_categorical']

# Map each selected categorical to its encoded column name
cat_encoded_map = {
    'Tenure Type'  : 'Tenure Type Encoded',
    'Land Title'   : 'Land Title Encoded',
    'Property Type': 'Property Type Encoded',
}

# Replace Completion Year with engineered Property Age
numeric_features = [c for c in selected_numeric if c != 'Completion Year'] + ['Property Age']

# Replace raw categoricals with their encoded versions
encoded_categoricals = [cat_encoded_map[c] for c in selected_categorical if c in cat_encoded_map]

# Add all engineered features and state dummies
FEATURES = (
    numeric_features +
    encoded_categoricals +
    ['Transit Zone', 'Facility Count', 'Facility Tier'] +
    list(state_dummies.columns)
)

X = df[FEATURES].copy()
y = df['price'].astype('Int64').astype(float)

assert X.isnull().sum().sum() == 0, "Null values found in feature matrix"
print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} features, 0 nulls")
print(f"Target range  : RM {int(y.min()):,} – RM {int(y.max()):,}, mean RM {int(y.mean()):,}\n")


# ── Step 8: Train/Test split (80/20) ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

assert len(set(X_train.index) & set(X_test.index)) == 0, "Index overlap detected"
print(f"Train set: {X_train.shape[0]} rows  |  Test set: {X_test.shape[0]} rows  (80/20, seed=42)")
print(f"Train mean: RM {int(y_train.mean()):,}  |  Test mean: RM {int(y_test.mean()):,}  (consistent distribution)\n")


# ── Step 9: Save outputs ──────────────────────────────────────────────────────
X_train.to_csv('data/X_train.csv', index=False)
X_test.to_csv('data/X_test.csv',   index=False)
y_train.to_csv('data/y_train.csv', index=False)
y_test.to_csv('data/y_test.csv',   index=False)

out = X.copy()
out['price'] = y.values
out.to_csv('data/houses_engineered.csv', index=False)

print(f"Saved: X_train ({X_train.shape}), X_test ({X_test.shape}), y_train, y_test, houses_engineered")
print(f"Next : run model_training.py")