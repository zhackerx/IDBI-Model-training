# ============================================================
# MSME Loan Risk Assessment - Model Training & Evaluation
# Models: XGBoost + LightGBM + Logistic Regression
# Includes: SHAP Explainability + Full Metrics Report
# ============================================================
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import warnings
warnings.filterwarnings('ignore')

# ML Models
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Evaluation Metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve, ConfusionMatrixDisplay
)

print("✅ All libraries imported successfully!")

# ============================================================
# STEP 1: LOAD CLEANED TRAIN/TEST DATA
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 1: LOADING DATA")
print("=" * 60)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 2. Build the correct paths for all 4 files
x_train_path = os.path.join(BASE_DIR, 'X_train.csv')
x_test_path  = os.path.join(BASE_DIR, 'X_test.csv')
y_train_path = os.path.join(BASE_DIR, 'y_train.csv')
y_test_path  = os.path.join(BASE_DIR, 'y_test.csv')

# 3. Load the datasets safely
X_train = pd.read_csv(x_train_path)
X_test  = pd.read_csv(x_test_path)
y_train = pd.read_csv(y_train_path).squeeze()  # Convert to Series
y_test  = pd.read_csv(y_test_path).squeeze()   # Convert to Series



print(f"✅ X_train : {X_train.shape}")
print(f"✅ X_test  : {X_test.shape}")
print(f"✅ y_train : {y_train.shape} | Distribution: {dict(y_train.value_counts())}")
print(f"✅ y_test  : {y_test.shape}  | Distribution: {dict(y_test.value_counts())}")

# ============================================================
# STEP 2: DEFINE ALL MODELS
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 2: DEFINING MODELS")
print("=" * 60)

models = {

    'Logistic Regression': LogisticRegression(
        max_iter      = 1000,
        random_state  = 42,
        class_weight  = 'balanced'
    ),

    'Random Forest': RandomForestClassifier(
        n_estimators  = 200,
        max_depth     = 10,
        random_state  = 42,
        class_weight  = 'balanced',
        n_jobs        = -1
    ),

    'XGBoost': XGBClassifier(
        n_estimators        = 300,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        scale_pos_weight    = (y_train == 0).sum() / (y_train == 1).sum(),
        use_label_encoder   = False,
        eval_metric         = 'logloss',
        random_state        = 42,
        n_jobs              = -1
    ),

    'LightGBM': LGBMClassifier(
        n_estimators        = 300,
        max_depth           = 6,
        learning_rate       = 0.05,
        subsample           = 0.8,
        colsample_bytree    = 0.8,
        class_weight        = 'balanced',
        random_state        = 42,
        n_jobs              = -1,
        verbose             = -1
    )
}

print(f"✅ {len(models)} models defined:")
for name in models:
    print(f"   → {name}")

# ============================================================
# STEP 3: TRAIN ALL MODELS & COLLECT RESULTS
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 3: TRAINING ALL MODELS")
print("=" * 60)

results     = {}
trained     = {}

for name, model in models.items():
    print(f"\n🔄 Training {name}...")

    # Train
    model.fit(X_train, y_train)

    # Predict
    y_pred      = model.predict(X_test)
    y_prob      = model.predict_proba(X_test)[:, 1]

    # Metrics
    acc         = accuracy_score(y_test, y_pred)
    prec        = precision_score(y_test, y_pred)
    rec         = recall_score(y_test, y_pred)
    f1          = f1_score(y_test, y_pred)
    auc         = roc_auc_score(y_test, y_prob)

    results[name] = {
        'Accuracy'  : round(acc  * 100, 2),
        'Precision' : round(prec * 100, 2),
        'Recall'    : round(rec  * 100, 2),
        'F1 Score'  : round(f1   * 100, 2),
        'AUC-ROC'   : round(auc  * 100, 2)
    }

    trained[name] = {
        'model'  : model,
        'y_pred' : y_pred,
        'y_prob' : y_prob
    }

    print(f"   ✅ Accuracy  : {acc*100:.2f}%")
    print(f"   ✅ Precision : {prec*100:.2f}%")
    print(f"   ✅ Recall    : {rec*100:.2f}%")
    print(f"   ✅ F1 Score  : {f1*100:.2f}%")
    print(f"   ✅ AUC-ROC   : {auc*100:.2f}%")

# ============================================================
# STEP 4: MODEL COMPARISON TABLE
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 4: MODEL COMPARISON")
print("=" * 60)

results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('AUC-ROC', ascending=False)
print(results_df.to_string())

best_model_name = results_df.index[0]
print(f"\n🏆 Best Model : {best_model_name}")
print(f"   AUC-ROC   : {results_df.loc[best_model_name, 'AUC-ROC']}%")
print(f"   F1 Score  : {results_df.loc[best_model_name, 'F1 Score']}%")

# ============================================================
# STEP 5: PLOT MODEL COMPARISON BAR CHART
# ============================================================

print("\n📊 Plotting Model Comparison Chart...")

fig, ax = plt.subplots(figsize=(12, 6))

metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
x       = np.arange(len(metrics))
width   = 0.2
colors  = ['#009fe4', '#00bb7e', '#7d55c7', '#e6007c']

for i, (name, row) in enumerate(results_df.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(x + i * width, vals, width, label=name, color=colors[i], alpha=0.85)
    
    for bar, val in zip(bars, vals):
        # ✅ CORRECT — full ax.text() call on proper lines
        ax.text(
            bar.get_x() + bar.get_width() / 2,   # ← x position (center of bar)
            bar.get_height() + 0.5,               # ← y position (top of bar)
            f'{val:.1f}%',                        # ← label text
            ha         = 'center',
            va         = 'bottom',
            fontsize   = 7.5,
            fontweight = 'bold'
        )

ax.set_xlabel('Metrics', fontsize=12)
ax.set_ylabel('Score (%)', fontsize=12)
ax.set_title('Model Comparison – All Metrics', fontsize=14, fontweight='bold')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics, fontsize=11)
ax.set_ylim(50, 110)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150)
plt.show()
print("✅ Saved → model_comparison.png")


# ============================================================
# STEP 6: CONFUSION MATRIX FOR BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"📌 STEP 6: CONFUSION MATRIX — {best_model_name}")
print("=" * 60)

best_y_pred = trained[best_model_name]['y_pred']
cm          = confusion_matrix(y_test, best_y_pred)

fig, ax = plt.subplots(figsize=(7, 5))
disp    = ConfusionMatrixDisplay(
    confusion_matrix  = cm,
    display_labels    = ['Good Loan (0)', 'Bad Loan (1)']
)
disp.plot(ax=ax, colorbar=False, cmap='Blues')
ax.set_title(f'Confusion Matrix — {best_model_name}', fontsize=13, fontweight='bold')

# Annotate TP, TN, FP, FN
labels = [['TN', 'FP'], ['FN', 'TP']]
for i in range(2):
    for j in range(2):
        ax.text(
            j, i - 0.25,
            labels[i][j],
            ha='center', va='center',
            fontsize=13, color='red', fontweight='bold'
        )

plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

tn, fp, fn, tp = cm.ravel()
print(f"\n   True  Positives (TP) — Correctly predicted Bad  Loan : {tp}")
print(f"   True  Negatives (TN) — Correctly predicted Good Loan : {tn}")
print(f"   False Positives (FP) — Good Loan predicted as Bad    : {fp}  ← Type I Error")
print(f"   False Negatives (FN) — Bad  Loan predicted as Good   : {fn}  ← Type II Error (Dangerous!)")
print("✅ Saved → confusion_matrix.png")

# ============================================================
# STEP 7: ROC CURVE — ALL MODELS
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 7: ROC CURVE — ALL MODELS")
print("=" * 60)

fig, ax = plt.subplots(figsize=(9, 6))
colors  = ['#009fe4', '#00bb7e', '#7d55c7', '#e6007c']

for i, (name, data) in enumerate(trained.items()):
    fpr, tpr, _ = roc_curve(y_test, data['y_prob'])
    auc_val     = results[name]['AUC-ROC']
    ax.plot(fpr, tpr, color=colors[i], lw=2,
            label=f"{name} (AUC = {auc_val:.1f}%)")

ax.plot([0, 1], [0, 1], 'k--', lw=1.5, label='Random Classifier')
ax.fill_between([0, 1], [0, 1], alpha=0.05, color='gray')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curve — All Models', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curve.png', dpi=150)
plt.show()
print("✅ Saved → roc_curve.png")

# ============================================================
# STEP 8: CLASSIFICATION REPORT — BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"📌 STEP 8: CLASSIFICATION REPORT — {best_model_name}")
print("=" * 60)

print(classification_report(
    y_test,
    best_y_pred,
    target_names = ['Good Loan (0)', 'Bad Loan (1)']
))

# ============================================================
# STEP 9: FEATURE IMPORTANCE — BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print(f"📌 STEP 9: FEATURE IMPORTANCE — {best_model_name}")
print("=" * 60)

best_model = trained[best_model_name]['model']

# Get feature importances (works for XGBoost, LightGBM, RandomForest)
if hasattr(best_model, 'feature_importances_'):
    importance_df = pd.DataFrame({
        'Feature'    : X_train.columns,
        'Importance' : best_model.feature_importances_
    }).sort_values('Importance', ascending=False).head(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    bars = ax.barh(
        importance_df['Feature'],
        importance_df['Importance'],
        color='#009fe4',
        edgecolor='white',
        alpha=0.85
    )
    ax.invert_yaxis()
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top 20 Feature Importances — {best_model_name}',
                 fontsize=13, fontweight='bold')

    for bar, val in zip(bars, importance_df['Importance']):
        ax.text(
            bar.get_width() + 0.001,
            bar.get_y() + bar.get_height() / 2,
            f'{val:.4f}',
            va='center', fontsize=9
        )

    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=150)
    plt.show()
    print("✅ Saved → feature_importance.png")
    print("\n📋 Top 10 Most Important Features:")
    print(importance_df.head(10).to_string(index=False))

else:
    print("⚠️  Feature importance not available for Logistic Regression directly.")

# ============================================================
# STEP 10: SHAP EXPLAINABILITY
# ============================================================

print("\n" + "=" * 60)
print(f"📌 STEP 10: SHAP EXPLAINABILITY — {best_model_name}")
print("=" * 60)

# Use a sample of 500 rows for SHAP (faster computation)
X_shap_sample = X_test.sample(n=min(500, len(X_test)), random_state=42)

print("🔄 Computing SHAP values (this may take ~30–60 seconds)...")

# TreeExplainer works for XGBoost, LightGBM, RandomForest
explainer   = shap.TreeExplainer(best_model)
shap_values = explainer.shap_values(X_shap_sample)

# For binary classification, shap_values may be a list [class0```python
# For binary classification, shap_values may be a list [class0, class1]
# We take class 1 (Bad Loan prediction)
if isinstance(shap_values, list):
    shap_vals = shap_values[1]
else:
    shap_vals = shap_values

print("✅ SHAP values computed!")

# ── SHAP Summary Plot (Bar) ──────────────────────────────────
print("\n📊 Plotting SHAP Summary Bar Chart...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_shap_sample,
    plot_type = 'bar',
    max_display = 20,
    show = False
)
plt.title(f'SHAP Feature Importance (Bar) — {best_model_name}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → shap_summary_bar.png")

# ── SHAP Summary Plot (Beeswarm/Dot) ────────────────────────
print("\n📊 Plotting SHAP Beeswarm Plot...")
plt.figure(figsize=(10, 8))
shap.summary_plot(
    shap_vals,
    X_shap_sample,
    max_display = 20,
    show = False
)
plt.title(f'SHAP Beeswarm Plot — {best_model_name}',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → shap_beeswarm.png")

# ── SHAP Waterfall Plot (Single Prediction Explanation) ─────
print("\n📊 Plotting SHAP Waterfall (Single Applicant Explanation)...")
sample_idx  = 0   # First applicant in test sample
explanation = shap.Explanation(
    values        = shap_vals[sample_idx],
    base_values   = explainer.expected_value if not isinstance(
                        explainer.expected_value, list)
                    else explainer.expected_value[1],
    data          = X_shap_sample.iloc[sample_idx].values,
    feature_names = X_shap_sample.columns.tolist()
)

plt.figure(figsize=(12, 6))
shap.waterfall_plot(explanation, max_display=15, show=False)
plt.title('SHAP Waterfall — Why This Loan Was Approved/Rejected',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_waterfall.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → shap_waterfall.png")

# ── SHAP Force Plot (HTML — Interactive) ────────────────────
print("\n📊 Generating SHAP Force Plot (HTML)...")
shap.initjs()
force_plot = shap.force_plot(
    base_value    = explainer.expected_value if not isinstance(
                        explainer.expected_value, list)
                    else explainer.expected_value[1],
    shap_values   = shap_vals[0],
    features      = X_shap_sample.iloc[0],
    feature_names = X_shap_sample.columns.tolist()
)
shap.save_html('shap_force_plot.html', force_plot)
print("✅ Saved → shap_force_plot.html (open in browser for interactive view)")

# ============================================================
# STEP 11: PREDICT ON A NEW SINGLE APPLICANT
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 11: PREDICT NEW LOAN APPLICANT")
print("=" * 60)

# Load scaler
scaler = joblib.load(os.path.join(BASE_DIR, 'scaler.pkl'))

def predict_applicant(applicant_dict, model, scaler, feature_columns):
    """
    Predict loan risk for a single new applicant.

    Parameters:
        applicant_dict  : dict of feature_name → value
        model           : trained ML model
        scaler          : fitted StandardScaler
        feature_columns : list of column names used during training

    Returns:
        prediction      : 0 (Good) or 1 (Bad)
        probability     : probability of being a Bad Loan
        risk_band       : Green / Amber / Red
    """
    # Create DataFrame with same columns as training
    applicant_df = pd.DataFrame([applicant_dict])

    # Add any missing columns as 0 (e.g. one-hot encoded columns)
    for col in feature_columns:
        if col not in applicant_df.columns:
            applicant_df[col] = 0

    # Keep only training columns in correct order
    applicant_df = applicant_df[feature_columns]

    # Scale
    applicant_scaled = scaler.transform(applicant_df)

    # Predict
    prediction  = model.predict(applicant_scaled)[0]
    probability = model.predict_proba(applicant_scaled)[0][1]

    # Risk Band
    if probability < 0.35:
        risk_band = '🟢 GREEN  — Low Risk   → AUTO APPROVE'
    elif probability < 0.65:
        risk_band = '🟡 AMBER  — Medium Risk → MANUAL REVIEW'
    else:
        risk_band = '🔴 RED    — High Risk   → AUTO REJECT'

    return prediction, probability, risk_band


# ── Sample New Applicant ─────────────────────────────────────
# Fill values matching your encoded feature format
new_applicant = {
    'annual_income'           : 45000,
    'dti'                     : 18.5,
    'installment'             : 350.0,
    'int_rate'                : 14.5,
    'loan_amount'             : 15000,
    'total_acc'               : 8,
    'total_payment'           : 5000,
    'emp_length'              : 3,
    'grade'                   : 5,       # C = 5
    'sub_grade'               : 23,      # C3 = 23
    'home_ownership'          : 1,       # rent = 1
    'verification_status'     : 1,       # source verified = 1
    'application_type'        : 0,       # individual = 0
    'term'                    : 0,       # 36 months = 0
    'loan_age_months'         : 12,
    'issue_month'             : 3,
    'issue_year'              : 2021,
    'days_since_last_payment' : 90,
    'days_since_credit_pull'  : 30,
    'days_to_next_payment'    : 15,
    'loan_to_income_ratio'    : round(15000 / 45000, 4),
    'payment_ratio'           : round(5000  / 15000, 4),
    'emi_to_income_ratio'     : round(350   / (45000 / 12), 4),
    'address_state'           : 0.08,    # frequency encoded
    # purpose one-hot (set the relevant one to 1, rest 0)
    'purpose_car'             : 1,
    'purpose_credit_card'     : 0,
    'purpose_debt_consolidation': 0,
    'purpose_home_improvement': 0,
    'purpose_other'           : 0,
}

prediction, probability, risk_band = predict_applicant(
    applicant_dict  = new_applicant,
    model           = best_model,
    scaler          = scaler,
    feature_columns = X_train.columns.tolist()
)


print(f"\n{'='*55} ")
print(f"  🏦 LOAN RISK ASSESSMENT RESULT")
print(f"{'='*55}")
print(f"  Prediction   : {'❌ BAD LOAN (Reject)'  if prediction == 1 else '✅ GOOD LOAN (Approve)'}")
print(f"  Risk Score   : {probability*100:.2f}%  (probability of default)")
print(f"  Risk Band    : {risk_band}")
print(f"{'='*55}")

# ============================================================
# STEP 12: SHAP EXPLANATION FOR NEW APPLICANT
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 12: SHAP EXPLANATION FOR NEW APPLICANT")
print("=" * 60)

# Prepare applicant DataFrame
applicant_df = pd.DataFrame([new_applicant])
for col in X_train.columns:
    if col not in applicant_df.columns:
        applicant_df[col] = 0
applicant_df = applicant_df[X_train.columns]

# Scale
applicant_scaled_df = pd.DataFrame(
    scaler.transform(applicant_df),
    columns = X_train.columns
)

# SHAP values for this applicant
shap_single = explainer.shap_values(applicant_scaled_df)
if isinstance(shap_single, list):
    shap_single_vals = shap_single[1][0]
else:
    shap_single_vals = shap_single[0]

# Top factors pushing towards default
shap_series = pd.Series(shap_single_vals, index=X_train.columns)
top_risk_factors    = shap_series.nlargest(5)
top_safety_factors  = shap_series.nsmallest(5)

print("\n🔴 Top 5 Factors INCREASING Risk (pushing towards Default):")
for feat, val in top_risk_factors.items():
    print(f"   → {feat:<35} SHAP = +{val:.4f}")

print("\n🟢 Top 5 Factors DECREASING Risk (pushing towards Approval):")
for feat, val in top_safety_factors.items():
    print(f"   → {feat:<35} SHAP = {val:.4f}")

# Waterfall for new applicant
explanation_new = shap.Explanation(
    values        = shap_single_vals,
    base_values   = explainer.expected_value if not isinstance(
                        explainer.expected_value, list)
                    else explainer.expected_value[1],
    data          = applicant_scaled_df.iloc[0].values,
    feature_names = X_train.columns.tolist()
)

plt.figure(figsize=(12, 6))
shap.waterfall_plot(explanation_new, max_display=15, show=False)
plt.title('SHAP Explanation — New Applicant Decision Breakdown',
          fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_new_applicant.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Saved → shap_new_applicant.png")

# ============================================================
# STEP 13: FINANCIAL HEALTH CARD (5-Dimension Score)
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 13: FINANCIAL HEALTH CARD — NEW APPLICANT")
print("=" * 60)

def compute_health_card(applicant):
    """
    Compute 5-dimension Financial Health Score (0–100 each).
    Overall score = weighted average (0–1000 scale).
    """

    # ── 1. Liquidity Score ───────────────────────────────────
    # Based on: emi_to_income_ratio (lower = better)
    emi_ratio       = applicant.get('emi_to_income_ratio', 0.5)
    liquidity       = max(0, min(100, round((1 - emi_ratio) * 100)))

    # ── 2. Solvency Score ────────────────────────────────────
    # Based on: dti (lower = better), loan_to_income_ratio
    dti             = applicant.get('dti', 30)
    lti             = applicant.get('loan_to_income_ratio', 0.5)
    solvency        = max(0, min(100, round(100 - (dti * 1.5) - (lti * 20))))

    # ── 3. Growth Score ──────────────────────────────────────
    # Based on: annual_income bracket, loan_age_months
    income          = applicant.get('annual_income', 30000)
    growth          = max(0, min(100, round(min(income / 1000, 100))))

    # ── 4. Compliance Score ──────────────────────────────────
    # Based on: verification_status, grade
    verify          = applicant.get('verification_status', 0)   # 0,1,2
    grade           = applicant.get('grade', 3)                  # 1–7
    compliance      = max(0, min(100, round((verify / 2) * 50 + (grade / 7) * 50)))

    # ── 5. Repayment Score ───────────────────────────────────
    # Based on: payment_ratio, days_since_last_payment
    pay_ratio       = applicant.get('payment_ratio', 0)
    days_late       = applicant.get('days_since_last_payment', 30)
    repayment       = max(0, min(100, round(
        (pay_ratio * 60) + max(0, (1 - days_late / 365)) * 40
    )))

    # ── Overall Score (0–1000) ───────────────────────────────
    weights         = {
        'Liquidity'  : 0.25,
        'Solvency'   : 0.25,
        'Growth'     : 0.15,
        'Compliance' : 0.20,
        'Repayment'  : 0.15
    }
    scores = {
        'Liquidity'  : liquidity,
        'Solvency'   : solvency,
        'Growth'     : growth,
        'Compliance' : compliance,
        'Repayment'  : repayment
    }
    overall = round(sum(scores[k] * weights[k] for k in scores) * 10)

    return scores, overall


scores, overall = compute_health_card(new_applicant)

# print(f"\n  📊 MSME FINANCIAL HEALTH CARD")
# print(f"  {'─'*40}")
# for dim, score in scores.items():
#     bar     = '█' * (score // 10) + '░' * (10 - score // 10)
#     status  = '🟢' if score >= 70 else ('🟡' if score >= 40 else '🔴')
#     print(f"  {status} {dim:<12} : [{bar}] {score}/100")
# print(f"  {'─'*40}")
# print(f"  🏆 Overall Score : {overall} / 1000")

# if overall >= 700:
#     band = '🟢 STRONG  — Recommend Approval'
# elif overall >= 500:
#     band = '🟡 MODERATE — Conditional Approval'
# else:
#     band = '🔴 WEAK    — High Risk, Reject'

# print(f"  📌 Risk Band     : {band}")
# print(f"  {'─'*40}")

# # ── Radar Chart for Health Card ──────────────────────────────
# print("\n📊 Plotting Financial Health Card Radar Chart...")

# categories  = list(scores.keys())
# values      = list(scores.values())
# values     += values[:1]   # close the polygon

# # ✅ CORRECT LINE (was cut off earlier)
# angles      = [n / float(len(categories)) * 2 * np.pi for n in range(len(categories))]
# angles     += angles[:1]   # close the polygon

# fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))

# # Draw the chart
# ax.plot(angles, values, color='#009fe4', linewidth=2, linestyle='solid')
# ax.fill(angles, values, color='#009fe4', alpha=0.25)

# # Add dimension labels
# ax.set_xticks(angles[:-1])
# ax.set_xticklabels(categories, fontsize=12, fontweight='bold')

# # Add score rings
# ax.set_yticks([20, 40, 60, 80, 100])
# ax.set_yticklabels(['20', '40', '60', '80', '100'], fontsize=8, color='grey')
# ax.set_ylim(0, 100)

# # Add score values on the chart
# for angle, value, label in zip(angles[:-1], values[:-1], categories):
#     ax.text(
#         angle, value + 8,
#         str(value),
#         ha='center', va='center',
#         fontsize=11, fontweight='bold', color='#009fe4'
#     )

# ax.set_title(
#     f'Financial Health Card\nOverall Score: {overall}/1000  |  {band}',
#     fontsize=12, fontweight='bold', pad=20
# )
# ax.grid(color='grey', linestyle='--', linewidth=0.5, alpha=0.5)

# plt.tight_layout()
# plt.savefig('financial_health_card.png', dpi=150, bbox_inches='tight')
# plt.show()
# print("✅ Saved → financial_health_card.png")

# ============================================================
# STEP 14: SAVE BEST MODEL
# ============================================================

print("\n" + "=" * 60)
print("📌 STEP 14: SAVING BEST MODEL")
print("=" * 60)

joblib.dump(best_model, 'best_model.pkl')
print(f"✅ Best model saved → best_model.pkl  ({best_model_name})")

# Save model metadata
model_meta = {
    'model_name'      : best_model_name,
    'auc_roc'         : results_df.loc[best_model_name, 'AUC-ROC'],
    'f1_score'        : results_df.loc[best_model_name, 'F1 Score'],
    'accuracy'        : results_df.loc[best_model_name, 'Accuracy'],
    'features'        : X_train.columns.tolist(),
    'n_features'      : X_train.shape[1],
    'train_samples'   : X_train.shape[0],
    'test_samples'    : X_test.shape[0],
}
joblib.dump(model_meta, 'model_metadata.pkl')
print("✅ Model metadata saved → model_metadata.pkl")

# ============================================================
# STEP 15: LOAD & USE MODEL IN PRODUCTION (Demo)
# ============================================================

# print("\n" + "=" * 60)
# print("📌 STEP 15: PRODUCTION USAGE DEMO")
# print("=" * 60)

# # This is how you load and use the model later
# loaded_model    = joblib.load('best_model.pkl')
# loaded_scaler   = joblib.load('scaler.pkl')
# loaded_meta     = joblib.load('model_metadata.pkl')

# print(f"✅ Loaded Model    : {loaded_meta['model_name']}")
# print(f"✅ AUC-ROC         : {loaded_meta['auc_roc']}%")
# print(f"✅ Total Features  : {loaded_meta['n_features']}")
# print(f"✅ Trained On      : {loaded_meta['train_samples']} samples")

# # Quick prediction using loaded model
# pred, prob, band = predict_applicant(
#     applicant_dict  = new_applicant,
#     model           = loaded_model,
#     scaler          = loaded_scaler,
#     feature_columns = loaded_meta['features']
# )

# print(f"\n🔁 Re-prediction using loaded model:")
# print(f"   Result      : {'❌ REJECT' if pred == 1 else '✅ APPROVE'}")
# print(f"   Probability : {prob*100:.2f}%")
# print(f"   Risk Band   : {band}")

# ============================================================
# STEP 16: FINAL SUMMARY REPORT
# ============================================================

print("\n" + "=" * 60)
print("📋 COMPLETE PIPELINE SUMMARY REPORT")
print("=" * 60)

print(f"""
┌──────────────────────────────────────────────────────────┐
│         MSME LOAN RISK MODEL — FINAL REPORT              │
├──────────────────────────────────────────────────────────┤
│  DATASET                                                 │
│  ├─ Total Records     : ~38,000                          │
│  ├─ Train Samples     : {X_train.shape[0]:<6} (80%)                   │
│  └─ Test  Samples     : {X_test.shape[0]:<6} (20%)                   │
├──────────────────────────────────────────────────────────┤
│  MODELS TRAINED                                          │
│  ├─ Logistic Regression                                  │
│  ├─ Random Forest                                        │
│  ├─ XGBoost                                              │
│  └─ LightGBM                                             │
├──────────────────────────────────────────────────────────┤
│  BEST MODEL : {best_model_name:<42}│
│  ├─ Accuracy  : {results_df.loc[best_model_name, 'Accuracy']:<6}%                               │
│  ├─ Precision : {results_df.loc[best_model_name, 'Precision']:<6}%                               │
│  ├─ Recall    : {results_df.loc[best_model_name, 'Recall']:<6}%                               │
│  ├─ F1 Score  : {results_df.loc[best_model_name, 'F1 Score']:<6}%                               │
│  └─ AUC-ROC   : {results_df.loc[best_model_name, 'AUC-ROC']:<6}%                               │
├──────────────────────────────────────────────────────────┤
│  EXPLAINABILITY                                          │
│  ├─ SHAP Summary Bar      → shap_summary_bar.png        │
│  ├─ SHAP Beeswarm         → shap_beeswarm.png           │
│  ├─ SHAP Waterfall        → shap_waterfall.png          │
│  ├─ SHAP New Applicant    → shap_new_applicant.png      │
│  └─ SHAP Force Plot       → shap_force_plot.html        │
├──────────────────────────────────────────────────────────┤
│  FINANCIAL HEALTH CARD                                   │
│  ├─ Liquidity Score       : {scores['Liquidity']}/100                    │
│  ├─ Solvency Score        : {scores['Solvency']}/100                    │
│  ├─ Growth Score          : {scores['Growth']}/100                    │
│  ├─ Compliance Score      : {scores['Compliance']}/100                    │
│  ├─ Repayment Score       : {scores['Repayment']}/100                    │
│  └─ Overall Score         : {overall}/1000                  │
├──────────────────────────────────────────────────────────┤
│  SAVED FILES                                             │
│  ├─ best_model.pkl        → Trained```python
│  ├─ best_model.pkl        → Trained model                │
│  ├─ scaler.pkl            → Feature scaler               │
│  ├─ model_metadata.pkl    → Model info & feature list    │
│  ├─ model_comparison.png  → All models metric chart      │
│  ├─ confusion_matrix.png  → TP/TN/FP/FN breakdown        │
│  ├─ roc_curve.png         → ROC curves all models        │
│  ├─ feature_importance.png→ Top 20 features              │
│  ├─ financial_health_card.png → Radar chart              │
│  └─ shap_force_plot.html  → Interactive SHAP             │
└──────────────────────────────────────────────────────────┘
""")

print("🎉 Full Model Pipeline Complete! Ready for Deployment.\n")
