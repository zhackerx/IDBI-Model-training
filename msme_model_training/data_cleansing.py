# ============================================================
# MSME Loan Risk Assessment - Data Cleaning Pipeline
# Tailored to Exact Data Format (38K Records)
# ============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# STEP 1: DEFINE COLUMN NAMES & LOAD DATA
# ============================================================

COLUMN_NAMES = [
    'loan_id', 'address_state', 'application_type', 'emp_length',
    'emp_title', 'grade', 'home_ownership', 'issue_date',
    'last_credit_pull_date', 'last_payment_date', 'loan_status',
    'next_payment_date', 'member_id', 'purpose', 'sub_grade',
    'term', 'verification_status', 'annual_income', 'dti',
    'installment', 'int_rate', 'loan_amount', 'total_acc', 'total_payment'
]

# ── Load Excel ──────────────────────────────────────────────
# If your Excel already has a header row, use header=0
# If NO header row exists, use header=None and pass names=COLUMN_NAMES

df = pd.read_excel('../data/financial_loan.xlsx', header=0)  # Change to header=None if no header

# If no header, uncomment below:
# df = pd.read_excel('loan_data.xlsx', header=None, names=COLUMN_NAMES)

# Standardize column names (strip spaces, lowercase)
df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')

print("✅ Data Loaded!")
print(f"   Shape     : {df.shape}")
print(f"   Columns   : {list(df.columns)}\n")

# ============================================================
# STEP 2: QUICK SNAPSHOT
# ============================================================

print("=" * 60)
print("📌 FIRST 3 ROWS")
print("=" * 60)
print(df.head(3).to_string())

print("\n" + "=" * 60)
print("📌 DATA TYPES")
print("=" * 60)
print(df.dtypes)

print("\n" + "=" * 60)
print("📌 MISSING VALUES")
print("=" * 60)
missing_df = pd.DataFrame({
    'Missing Count' : df.isnull().sum(),
    'Missing %'     : (df.isnull().sum() / len(df) * 100).round(2)
})
print(missing_df[missing_df['Missing Count'] > 0])

# ============================================================
# STEP 3: DROP ID COLUMNS (No Predictive Value)
# ============================================================

df.drop(columns=['loan_id', 'member_id'], inplace=True, errors='ignore')
print("\n✅ Dropped: loan_id, member_id")

# ============================================================
# STEP 4: REMOVE DUPLICATES
# ============================================================

before = len(df)
df.drop_duplicates(inplace=True)
print(f"✅ Removed {before - len(df)} duplicate rows | Remaining: {len(df)}")

# ============================================================
# STEP 5: STANDARDIZE ALL TEXT COLUMNS
# ============================================================

obj_cols = df.select_dtypes(include='object').columns
for col in obj_cols:
    df[col] = df[col].astype(str).str.strip().str.lower()

print(f"✅ Stripped & lowercased {len(obj_cols)} text columns")

# ============================================================
# STEP 6: CLEAN emp_length → Numeric (0 to 10)
# ============================================================

emp_map = {
    '< 1 year' : 0,
    '1 year'   : 1,
    '2 years'  : 2,
    '3 years'  : 3,
    '4 years'  : 4,
    '5 years'  : 5,
    '6 years'  : 6,
    '7 years'  : 7,
    '8 years'  : 8,
    '9 years'  : 9,
    '10+ years': 10,
    'nan'      : np.nan
}

df['emp_length'] = df['emp_length'].map(emp_map)
print(f"✅ emp_length → numeric | Nulls after: {df['emp_length'].isnull().sum()}")

# ============================================================
# STEP 7: CLEAN term → Numeric (36 or 60)
# ============================================================

# Handles " 60 months" (leading space) and "60 months"
df['term'] = df['term'].astype(str).str.strip().str.extract(r'(\d+)').astype(float)
print(f"✅ term → numeric | Unique values: {sorted(df['term'].dropna().unique())}")

# ============================================================
# STEP 8: CLEAN int_rate & dti → Percentage Format
# ```python
# ============================================================
# STEP 8: CLEAN int_rate & dti → Percentage Format
# (Your data has decimal: 0.1527 = 15.27%)
# ============================================================

df['int_rate'] = pd.to_numeric(df['int_rate'], errors='coerce')
df['dti']      = pd.to_numeric(df['dti'],      errors='coerce')

# Convert to percentage if values are in decimal form (< 1.0)
if df['int_rate'].dropna().max() <= 1.0:
    df['int_rate'] = (df['int_rate'] * 100).round(4)
    print("✅ int_rate converted: decimal → percentage (e.g. 0.1527 → 15.27)")

if df['dti'].dropna().max() <= 1.0:
    df['dti'] = (df['dti'] * 100).round(4)
    print("✅ dti converted: decimal → percentage (e.g. 0.01 → 1.0)")

# ============================================================
# STEP 9: CONVERT NUMERIC COLUMNS
# ============================================================

num_cols = ['annual_income', 'installment', 'loan_amount',
            'total_acc', 'total_payment']

for col in num_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

print("✅ Numeric columns confirmed: annual_income, installment, loan_amount, total_acc, total_payment")

# ============================================================
# STEP 10: PARSE DATE COLUMNS (DD-MM-YYYY format)
# ============================================================

date_cols = ['issue_date', 'last_credit_pull_date',
             'last_payment_date', 'next_payment_date']

for col in date_cols:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], dayfirst=True, errors='coerce')

print("✅ All date columns parsed with dayfirst=True (DD-MM-YYYY)")

# Check parse success
for col in date_cols:
    if col in df.columns:
        null_count = df[col].isnull().sum()
        print(f"   → {col}: {null_count} unparsed (NaT)")

# ============================================================
# STEP 11: DATE FEATURE ENGINEERING
# ============================================================

reference_date = pd.Timestamp('2022-01-01')  # Use a fixed reference near your data range

if 'issue_date' in df.columns:
    df['loan_age_months']  = ((reference_date - df['issue_date']).dt.days / 30).round(1)
    df['issue_month']      = df['issue_date'].dt.month
    df['issue_year']       = df['issue_date'].dt.year

if 'last_payment_date' in df.columns:
    df['days_since_last_payment'] = (reference_date - df['last_payment_date']).dt.days

if 'last_credit_pull_date' in df.columns:
    df['days_since_credit_pull']  = (reference_date - df['last_credit_pull_date']).dt.days

if 'next_payment_date' in df.columns:
    df['days_to_next_payment']    = (df['next_payment_date'] - reference_date).dt.days

# Drop original date columns
df.drop(columns=date_cols, inplace=True, errors='ignore')
print("✅ Date features engineered | Raw date columns dropped")
print("   New columns: loan_age_months, issue_month, issue_year,")
print("                days_since_last_payment, days_since_credit_pull, days_to_next_payment")

# ============================================================
# STEP 12: DERIVED FEATURE ENGINEERING
# ============================================================

# Loan-to-Income Ratio
df['loan_to_income_ratio'] = (
    df['loan_amount'] / df['annual_income'].replace(0, np.nan)
).round(4)

# Payment Ratio (how much has been repaid vs loan amount)
df['payment_ratio'] = (
    df['total_payment'] / df['loan_amount'].replace(0, np.nan)
).round(4)

# EMI Stress Ratio (monthly EMI vs monthly income)
df['emi_to_income_ratio'] = (
    df['installment'] / (df['annual_income'].replace(0, np.nan) / 12)
).round(4)

print("✅ Derived features created:")
print("   → loan_to_income_ratio = loan_amount / annual_income")
print("   → payment_ratio        = total_payment / loan_amount")
print("   → emi_to_income_ratio  = installment / (annual_income / 12)")

# ============================================================
# STEP 13: TARGET VARIABLE ENCODING (loan_status)
# ============================================================

print("\n" + "=" * 60)
print("📌 TARGET VARIABLE: loan_status")
print("=" * 60)
print("Before encoding:")
print(df['loan_status'].value_counts())

status_map = {
    'fully paid'                                          : 0,
    'current'                                             : 0,
    'charged off'                                         : 1,
    'default'                                             : 1,
    'late (31-120 days)'                                  : 1,
    'late (16-30 days)'                                   : 1,
    'in grace period'                                     : 1,
    'does not meet the credit policy. status:fully paid'  : 0,
    'does not meet the credit policy. status:charged off' : 1
}

df['loan_status'] = df['loan_status'].map(status_map)

# Drop rows with unrecognized loan_status
before = len(df)
df.dropna(subset=['loan_status'], inplace=True)
df['loan_status'] = df['loan_status'].astype(int)
print(f"\nAfter encoding (dropped {before - len(df)} unknown rows):")
print(df['loan_status'].value_counts())
print(f"   0 = Good (Fully Paid / Current) : {(df['loan_status']==0).sum()} ({(df['loan_status']==0).mean()*100:.1f}%)")
print(f"   1 = Bad  (Charged Off / Default): {(df['loan_status']==1).sum()} ({(df['loan_status']==1).mean()*100:.1f}%)")

# ============================================================
# STEP 14: ENCODE grade (Ordinal: A=7 → G=1)
# ============================================================

grade_map = {'a': 7, 'b': 6, 'c': 5, 'd': 4, 'e': 3, 'f': 2, 'g': 1}
df['grade'] = df['grade'].map(grade_map)
print("\n✅ grade → ordinal encoded (a=7 to g=1)")
print(f"   Nulls after mapping: {df['grade'].isnull().sum()}")

# ============================================================
# STEP 15: ENCODE sub_grade (Ordinal: A1=35 → G5=1)
# ============================================================

grades    = ['a', 'b', 'c', 'd', 'e', 'f', 'g']
sub_list  = [f"{g}{n}" for g in grades for n in range(1, 6)]  # a1, a2 ... g5
sub_map   = {sg: (len(sub_list) - i) for i, sg in enumerate(sub_list)}
df['sub_grade'] = df['sub_grade'].map(sub_map)
print(f"✅ sub_grade → ordinal encoded (a1=35 to g5=1)")
print(f"   Nulls after mapping: {df['sub_grade'].isnull().sum()}")

# ============================================================
# STEP 16: ENCODE home_ownership (Ordinal)```python
# ============================================================
# STEP 16: ENCODE home_ownership (Ordinal)
# ============================================================

home_map = {
    'own'      : 3,
    'mortgage' : 2,
    'rent'     : 1,
    'other'    : 0,
    'none'     : 0,
    'any'      : 0,
    'nan'      : np.nan
}

df['home_ownership'] = df['home_ownership'].map(home_map)
print("✅ home_ownership → ordinal (own=3, mortgage=2, rent=1, other/none=0)")
print(f"   Nulls after mapping: {df['home_ownership'].isnull().sum()}")

# ============================================================
# STEP 17: ENCODE verification_status (Ordinal)
# ============================================================

verify_map = {
    'verified'        : 2,
    'source verified' : 1,
    'not verified'    : 0,
    'nan'             : np.nan
}

df['verification_status'] = df['verification_status'].map(verify_map)
print("✅ verification_status → ordinal (verified=2, source verified=1, not verified=0)")
print(f"   Nulls after mapping: {df['verification_status'].isnull().sum()}")

# ============================================================
# STEP 18: ENCODE application_type (Binary)
# ============================================================

app_map = {
    'individual' : 0,
    'joint app'  : 1,
    'direct_pay' : 0,
    'nan'        : np.nan
}

df['application_type'] = df['application_type'].map(app_map)
print("✅ application_type → binary (individual=0, joint app=1)")
print(f"   Nulls after mapping: {df['application_type'].isnull().sum()}")

# ============================================================
# STEP 19: ENCODE term (Binary)
# ============================================================

term_map = {36.0: 0, 60.0: 1}
df['term'] = df['term'].map(term_map)
print("✅ term → binary (36 months=0, 60 months=1)")
print(f"   Nulls after mapping: {df['term'].isnull().sum()}")

# ============================================================
# STEP 20: ENCODE purpose (One-Hot Encoding)
# ============================================================

print(f"\n📌 purpose unique values ({df['purpose'].nunique()}):")
print(df['purpose'].value_counts())

purpose_dummies = pd.get_dummies(df['purpose'], prefix='purpose', drop_first=True)
df = pd.concat([df, purpose_dummies], axis=1)
df.drop(columns=['purpose'], inplace=True)
print(f"✅ purpose → one-hot encoded | {purpose_dummies.shape[1]} new columns added")

# ============================================================
# STEP 21: ENCODE address_state (Frequency Encoding)
# ============================================================

freq_map = df['address_state'].value_counts(normalize=True).to_dict()
df['address_state'] = df['address_state'].map(freq_map).round(5)
print("✅ address_state → frequency encoded (proportion of each state in dataset)")
print(f"   Sample: CA={freq_map.get('ca', 0):.4f}, GA={freq_map.get('ga', 0):.4f}")

# ============================================================
# STEP 22: ENCODE emp_title (Top 20 + 'other' → One-Hot)
# ============================================================

print(f"\n📌 emp_title unique values before grouping: {df['emp_title'].nunique()}")

top_titles      = df['emp_title'].value_counts().nlargest(20).index.tolist()
df['emp_title'] = df['emp_title'].apply(lambda x: x if x in top_titles else 'other')

title_dummies   = pd.get_dummies(df['emp_title'], prefix='emp_title', drop_first=True)
df              = pd.concat([df, title_dummies], axis=1)
df.drop(columns=['emp_title'], inplace=True)
print(f"✅ emp_title → grouped to Top 20 + 'other', one-hot encoded")
print(f"   {title_dummies.shape[1]} new emp_title columns added")

# ============================================================
# STEP 23: HANDLE MISSING VALUES
# ============================================================

print("\n" + "=" * 60)
print("📌 MISSING VALUE IMPUTATION")
print("=" * 60)

# Separate target before imputation
target = df['loan_status'].copy()
df.drop(columns=['loan_status'], inplace=True)

# Numeric columns → median imputation
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in num_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"   → {col}: filled {null_count} nulls with median = {median_val:.4f}")

# Categorical columns (if any remain) → mode imputation
cat_cols = df.select_dtypes(include='object').columns.tolist()
for col in cat_cols:
    null_count = df[col].isnull().sum()
    if null_count > 0:
        mode_val = df[col].mode()[0]
        df[col].fillna(mode_val, inplace=True)
        print(f"   → {col}: filled {null_count} nulls with mode = {mode_val}")

# Reattach target
df['loan_status'] = target
print(f"\n✅ Imputation complete | Remaining nulls: {df.isnull().sum().sum()}")

# ============================================================
# STEP 24: OUTLIER TREATMENT (IQR Capping)
# ============================================================

print("\n" + "=" * 60)
print("📌 OUTLIER TREATMENT (IQR Capping)")
print("=" * 60)

outlier_cols = [
    'annual_income', 'loan_amount', 'dti', 'installment',
    'total_payment', 'int_rate', 'loan_to_income_ratio',
    'emi_to_income_ratio', 'payment_ratio',
    'days_since_last_payment', 'days_since_credit_pull',
    'loan_age_months'
]

for col in outlier_cols:
    if col in df.columns:
        Q1    = df[col].quantile(0.25)
        Q3    = df[col].quantile(0.75)
        IQR   = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        count = ((df[col] < lower) | (df[col] > upper)).sum()
        df[col] = df[col].clip(lower=lower, upper=upper)
        print(f"   → {col}: {count} outliers capped | Range [{lower:.2f}, {upper:.2f}]")

print("✅ Outlier treatment complete")

# ============================================================
# STEP 25: CONVERT BOOLEAN COLUMNS TO INT (from get_dummies)
# ============================================================

bool_cols = df.select_dtypes(include='bool').columns.tolist()
if bool_cols:
    df[bool_cols] = df[bool_cols].astype(int)
    print(f"\n✅ Converted {len(bool_cols)} boolean columns to int (0/1)")

# ============================================================
# STEP 26: FINAL VALIDATION CHECK
# ============================================================

print("\n" + "=" * 60)
print("📌 FINAL VALIDATION")
print("=" * 60)
print(f"   Final Shape          : {df.shape}")
print(f"   Total Nulls          : {df.isnull().sum().sum()}")
print(f"   Duplicate Rows       : {df.duplicated().sum()}")
print(f"   Total Features       : {df.shape[1] - 1}")
print(f"   Target Column        : loan_status")
print(f"\n📊 Target Distribution:")
print(f"   Good Loans (0)       : {(df['loan_status']==0).sum()} ({(df['loan_status']==0).mean()*100:.1f}%)")
print(f"   Bad  Loans (1)       : {(df['loan_status']==1).sum()} ({(df['loan_status']==1).mean()*100:.1f}%)")

print("\n📋 All Final Columns:")
for i, col in enumerate(df.columns, 1):
    print(f"   {i:3}. {col}")

# ============================================================
# STEP 27: SPLIT FEATURES & TARGET
# ============================================================

X = df.drop(columns=['loan_status'])
y = df['loan_status']

print(f"\n✅ X (Features) shape : {X.shape}")
print(f"✅ y (Target)   shape : {y.shape}")

# ============================================================
# STEP 28: HANDLE CLASS IMBALANCE USING SMOTE
# ============================================================

print("\n" + "=" * 60)
print("📌 CLASS IMBALANCE CHECK & SMOTE")
print("=" * 60)

imbalance_ratio = (y == 0).sum() / (y == 1).sum()
print(f"   Imbalance Ratio (Good:Bad) = {imbalance_ratio:.2f}:1")

if imbalance_ratio > 1.5:
    print("   ⚠️  Imbalance detected — applying SMOTE...")
    try:
        from imblearn.over_sampling import SMOTE
        smote            = SMOTE(random_state=42)
        X_res, y_res     = smote.fit_resample(X, y)
        print(f"   ✅ Before SMOTE → {dict(y.value_counts())}")
        print(f"   ✅ After  SMOTE → {dict(pd.Series(y_res).value_counts())}")
    except ImportError:
        print("   ❌ imbalanced-learn not installed.")
        print("      Run: pip install imbalanced-learn")
        print("   ⚠️  Proceeding WITHOUT SMOTE — using class_weight in model instead.")
        X_res, y_res = X, y
else:
    print("   ✅ Classes are balanced — SMOTE not needed.")
    X_res, y_res = X, y

# ============================================================
# STEP 29: FEATURE SCALING (StandardScaler)
# ============================================================

print("\n" + "=" * 60)
print("📌 FEATURE SCALING")
print("=" * 60)

scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X_res)
X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

print("✅ StandardScaler applied to all features")
print(f"   Mean  (sample): {X_scaled.iloc[:, :3].mean().round(4).to_dict()}")
print(f"   Std   (sample): {X_scaled.iloc[:, :3].std().round(4).to_dict()}")

# ============================================================
# STEP 30: TRAIN-TEST SPLIT (80/20 Stratified)
# ============================================================

print("\n" + "=" * 60)
print("📌 TRAIN-TEST SPLIT")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled,
    y_res,
    test_size    = 0.2,
    random_state = 42,
    stratify     = y_res
)

print(f"✅ Train Set : X={X_train.shape} | y={y_train.shape}")
print(f"✅ Test  Set : X={X_test.shape}  | y={y_test.shape}")
print(f"\n   Train Target Distribution : {dict(pd.Series(y_train).value_counts())}")
print(f"   Test  Target Distribution : {dict(pd.Series(y_test).value_counts())}")

# ============================================================
# STEP 31: SAVE ALL OUTPUTS
# ============================================================

print("\n" + "=" * 60)
print("📌 SAVING FILES")
print("=" * 60)

# Full cleaned dataset (pre-SMOTE)
df.to_csv('loan_data_cleaned.csv', index=False)
print("✅ loan_data_cleaned.csv       → Full cleaned dataset")

# Train / Test splits
X_train.to_csv('X_train.csv', index=False)
X_test.to_csv('X_test.csv',   index=False)
pd.Series(y_train).to_csv('y_train.csv', index=False, header=True)
pd.Series(y_test).to_csv('y_test.csv',   index=False, header=True)
print("✅ X_train.csv, X_test.csv     → Feature splits")
print("✅ y_train.csv, y_test.csv     → Target splits")

# Save scaler for later use in production
import joblib
joblib.dump(scaler, 'scaler.pkl')
print("✅ scaler.pkl                  → Saved StandardScaler for deployment")

# ============================================================
# STEP 32: FINAL SUMMARY REPORT
# ============================================================

print("\n" + "=" * 60)
print("📋 COMPLETE PIPELINE SUMMARY")
print("=" * 60)
print(f"""
┌──────────────────────────────────────────────────────┐
│           MSME LOAN DATA CLEANING REPORT             │
├──────────────────────────────────────────────────────┤
│  Raw Records Loaded         : ~38,000                │
│  Duplicates Removed         : ✅                     │
│  ID Columns Dropped         : loan_id, member_id     │
├──────────────────────────────────────────────────────┤
│  CLEANING                                            │
│  ├─ emp_length  : string → numeric (0–10)            │
│  ├─ term        : "60 months" → numeric (0/1)        │
│  ├─ int_rate    : decimal → percentage               │
│  ├─ dti         : decimal → percentage               │
│  └─ dates       : DD-MM-YYYY → datetime              │
├──────────────────────────────────────────────────────┤
│  ENCODING                                            │
│  ├─ grade           : ordinal (a=7 → g=1)            │
│  ├─ sub_grade       : ordinal (a1=35 → g5=1)         │
│  ├─ home_ownership  : ordinal (own=3 → other=0)      │
│  ├─ verification    : ordinal (verified=2 → 0)       │
│  ├─ application     : binary  (individual=0)         │
│  ├─ term            : binary  (36m=0, 60m=1)         │
│  ├─ purpose         : one-hot encoded                │
│  ├─ address_state   : frequency encoded              │
│  └─ emp_title       : top-20 + one-hot               │
├──────────────────────────────────────────────────────┤
│  FEATURE ENGINEERING                                 │
│  ├─ loan_age_months                                  │
│  ├─ issue_month / issue_year                         │
│  ├─ days_since_last_payment                          │
│  ├─ days_since_credit_pull                           │
│  ├─ days_```python
│  ├─ days_to_next_payment                            │
│  ├─ loan_to_income_ratio                            │
│  ├─ payment_ratio                                   │
│  └─ emi_to_income_ratio                             │
├──────────────────────────────────────────────────────┤
│  MISSING VALUES   : Median (numeric) / Mode (cat)   │
│  OUTLIERS         : IQR Capping (1.5x)              │
│  CLASS IMBALANCE  : SMOTE Applied                   │
│  SCALING          : StandardScaler                  │
│  TRAIN/TEST SPLIT : 80% / 20% Stratified            │
├──────────────────────────────────────────────────────┤
│  TARGET ENCODING                                     │
│  ├─ 0 = Good (Fully Paid / Current)                 │
│  └─ 1 = Bad  (Charged Off / Default / Late)         │
├──────────────────────────────────────────────────────┤
│  OUTPUT FILES                                        │
│  ├─ loan_data_cleaned.csv                           │
│  ├─ X_train.csv / X_test.csv                        │
│  ├─ y_train.csv / y_test.csv                        │
│  └─ scaler.pkl                                      │
└──────────────────────────────────────────────────────┘
""")

print("🎉 Pipeline Complete! Data is ready for Model Training.\n")
