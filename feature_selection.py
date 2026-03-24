"""
Feature Selection Script — houses_clean.csv
Person A: Data & Model Architect
Step 2 of pipeline: Statistically validate which features to keep before engineering.

Stages:
    1. Domain filter       — remove admin/ID/noise columns
    2. Variance threshold  — drop near-zero variance features
    3. Pearson + ANOVA     — correlation with price
    4. VIF                 — multicollinearity check
    5. Mutual Information  — linear + non-linear signal
    6. Preliminary RF      — model-based importance ranking
    7. Final decision      — consolidate all evidence
"""

import json
import pandas as pd
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from statsmodels.stats.outliers_influence import variance_inflation_factor

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv('data/houses_clean.csv')
df['price_num'] = df['price'].astype('Int64').astype(float)
candidates = [c for c in df.columns if c not in ('price', 'price_num')]

print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1] - 1} candidate features\n")


# ── Stage 1: Domain filter ─────────────────────────────────────────────────────
domain_drop = [
    'Ad List', 'Building Name', 'Developer', 'Address',
    'Firm Type', 'Firm Number', 'REN Number', 'Park', 'School', 'Facilities'
]
remaining = [c for c in candidates if c not in domain_drop]

print(f"Stage 1 — Domain filter")
print(f"  Dropped {len(domain_drop)} columns (IDs, agent info, free-text, over-imputed)")
print(f"  Remaining: {remaining}\n")


# ── Stage 2: Variance threshold ────────────────────────────────────────────────
le_tmp = LabelEncoder()
low_var = []
for col in remaining:
    vals = le_tmp.fit_transform(df[col].astype(str)) if df[col].dtype == object or str(df[col].dtype) == 'str' else df[col]
    if np.std(vals) < 0.01:
        low_var.append(col)

print(f"Stage 2 — Variance threshold (std < 0.01)")
if low_var:
    print(f"  Dropped: {low_var}")
else:
    print(f"  All {len(remaining)} features pass — no near-constant columns found")
remaining = [c for c in remaining if c not in low_var]
print()


# ── Stage 3: Pearson + ANOVA ───────────────────────────────────────────────────
numeric_cols = [c for c in remaining if str(df[c].dtype) not in ('object', 'str') and df[c].dtype != object]
cat_cols     = [c for c in remaining if c not in numeric_cols]

print(f"Stage 3 — Pearson correlation (numeric) + ANOVA (categorical)")
print(f"  {'Feature':<22} {'Statistic':>12}  {'p-value':>10}  Strength")
print(f"  {'-'*58}")

pearson_r = {}
for col in numeric_cols:
    r, p = stats.pearsonr(df[col].astype(float), df['price_num'])
    strength = 'Strong' if abs(r) > 0.30 else ('Moderate' if abs(r) > 0.10 else 'Weak')
    pearson_r[col] = abs(r)
    print(f"  {col:<22}  r = {r:>7.4f}  {p:.2e}  {strength}")

anova_f = {}
for col in cat_cols:
    groups = [df[df[col] == v]['price_num'].values for v in df[col].unique() if len(df[df[col] == v]) > 1]
    f, p = stats.f_oneway(*groups)
    anova_f[col] = f
    print(f"  {col:<22}  F = {f:>7.2f}  {p:.2e}  {'Significant' if p < 0.05 else 'Not significant'}")
print()


# ── Stage 4: VIF ──────────────────────────────────────────────────────────────
le_vif = LabelEncoder()
df_vif = pd.DataFrame({
    col: (le_vif.fit_transform(df[col].astype(str)) if df[col].dtype == object or str(df[col].dtype) == 'str' else df[col].astype(float))
    for col in remaining
})
vif_scores = {col: variance_inflation_factor(df_vif.values, i) for i, col in enumerate(df_vif.columns)}
high_vif   = [c for c, v in vif_scores.items() if v > 10]

print(f"Stage 4 — Variance Inflation Factor (VIF > 10 = high multicollinearity)")
for col, vif in sorted(vif_scores.items(), key=lambda x: -x[1]):
    flag = ' ← HIGH' if vif > 10 else (' ← MODERATE' if vif > 5 else '')
    print(f"  {col:<22}  VIF = {vif:>7.2f}{flag}")
print(f"  Note: High-VIF features retained — RF and XGBoost handle correlated features natively\n")


# ── Stage 5: Mutual Information ────────────────────────────────────────────────
le_mi = LabelEncoder()
df_mi = pd.DataFrame({
    col: (le_mi.fit_transform(df[col].astype(str)) if df[col].dtype == object or str(df[col].dtype) == 'str' else df[col].astype(float))
    for col in remaining
})
mi_scores = dict(zip(remaining, mutual_info_regression(df_mi, df['price_num'], random_state=42)))

print(f"Stage 5 — Mutual Information (linear + non-linear signal)")
print(f"  {'Feature':<22}  {'MI Score':>8}  Signal")
print(f"  {'-'*45}")
mi_drop = []
for col, score in sorted(mi_scores.items(), key=lambda x: -x[1]):
    strength = 'Strong' if score > 0.20 else ('Moderate' if score > 0.05 else 'Very weak')
    flag = ' ← drop candidate' if score < 0.05 else ''
    print(f"  {col:<22}  {score:>8.4f}  {strength}{flag}")
    if score < 0.05:
        mi_drop.append(col)
print()


# ── Stage 6: Preliminary Random Forest importance ──────────────────────────────
rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=42, n_jobs=-1)
rf.fit(df_mi, df['price_num'])
rf_scores = dict(zip(remaining, rf.feature_importances_))

print(f"Stage 6 — Preliminary Random Forest importance (max_depth=8, selection tool only)")
print(f"  {'Feature':<22}  {'Importance':>10}  Signal")
print(f"  {'-'*48}")
rf_drop = []
for col, imp in sorted(rf_scores.items(), key=lambda x: -x[1]):
    strength = 'High' if imp > 0.05 else ('Moderate' if imp > 0.01 else 'Low')
    flag = ' ← drop candidate' if imp < 0.01 else ''
    print(f"  {col:<22}  {imp:>10.4f}  {strength}{flag}")
    if imp < 0.01:
        rf_drop.append(col)
print()


# ── Stage 7: Final decision ────────────────────────────────────────────────────
# Drop rule: MI < 0.05 AND RF importance < 0.01 → statistical drop candidate
# Domain override: Tenure Type + Land Title are retained despite weak MI/RF
#   because ANOVA is significant (p < 0.05) and Part 1 documents them as
#   Malaysia-specific legal factors (Freehold premium, Bumi/Non-Bumi restrictions).
DOMAIN_KEEP = {'Tenure Type', 'Land Title'}

decisions = {}
for col in remaining:
    mi_val = mi_scores[col]
    rf_val = rf_scores[col]
    if mi_val < 0.05 and rf_val < 0.01 and col not in DOMAIN_KEEP:
        decisions[col] = 'DROP'
    else:
        decisions[col] = 'KEEP'

selected = [c for c, v in decisions.items() if v == 'KEEP']
dropped  = [c for c, v in decisions.items() if v == 'DROP']

# Build corr string from already-computed pearson_r / anova_f dicts
def corr_label(col):
    if col in pearson_r:
        r_raw = next(r for c, (r, _) in
                     [(c, stats.pearsonr(df[c].astype(float), df['price_num']))
                      for c in numeric_cols] if c == col)
        return f"r={pearson_r[col]*np.sign(r_raw):.2f}"
    return f"F={anova_f[col]:.1f}"

# Re-compute signed Pearson r for display (pearson_r stores absolute values)
signed_r = {col: stats.pearsonr(df[col].astype(float), df['price_num'])[0]
            for col in numeric_cols}

print(f"Stage 7 — Final decision")
print(f"  {'Feature':<22}  {'Pearson/ANOVA':>13}  {'MI':>6}  {'RF':>6}  Verdict")
print(f"  {'-'*65}")

for col in remaining:
    if col in numeric_cols:
        corr_str = f"r={signed_r[col]:>+.2f}"
    else:
        corr_str = f"F={anova_f[col]:.1f}"
    mi_str  = f"{mi_scores[col]:.3f}"
    rf_str  = f"{rf_scores[col]:.3f}"
    verdict = decisions[col]
    note    = '*' if col in DOMAIN_KEEP else ''
    print(f"  {col:<22}  {corr_str:>13}  {mi_str:>6}  {rf_str:>6}  {verdict}{note}")

print(f"""
  * Retained on domain justification: ANOVA confirms significant price difference
    between groups (p < 0.05), and Part 1 documents these as Malaysia-specific
    legal factors (Freehold premium, Bumi/Non-Bumi restrictions).

  Dropped features: MI < 0.05 and RF importance < 0.01 with no domain override.
""")

print(f"  Selected ({len(selected)}): {selected}")
print(f"  Dropped  ({len(dropped)}):  {dropped}")


# ── Save result ────────────────────────────────────────────────────────────────
result = {
    'selected_numeric':     [c for c in selected if c in numeric_cols],
    'selected_categorical': [c for c in selected if c in cat_cols],
    'dropped':              dropped + domain_drop,
}
with open('data/feature_selection_result.json', 'w') as f:
    json.dump(result, f, indent=2)

print(f"\nSaved: data/feature_selection_result.json")
print(f"Next : run feature_engineering.py")