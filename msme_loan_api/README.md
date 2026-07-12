# 🏦 MSME Loan Risk Assessment API

## Quick Start

### 1. Install dependencies
pip install -r requirements.txt

### 2. Place model files
Copy these into the model/ directory:
- best_model.pkl
- scaler.pkl
- model_metadata.pkl

### 3. Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

### 4. Open Swagger UI
http://localhost:8000/docs

### 5. Open Frontend Dashboard
http://localhost:8000/dashboard

---

## New Agentic RAG Endpoints (MVP)

- POST /rag/policies/reload
- POST /rag/analyze  (multipart/form-data with PDF file)
- GET  /rag/health

The RAG flow uses:
- PDF parsing (PyMuPDF)
- Chunking + local TF-IDF retrieval
- LangGraph workflow orchestration
- Corrective retrieval retry
- Analysis Agent + Judge Agent

### Vector DB Mode (Chroma)

By default, retrieval uses local TF-IDF (no external setup required).

To enable Chroma vector DB retrieval:

1. Install dependencies from requirements.txt (includes chromadb)
2. Start API with env vars:

Windows PowerShell:

```powershell
$env:RAG_BACKEND="chroma"
$env:CHROMA_PERSIST_DIR="./chroma_db"
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

macOS/Linux:

```bash
export RAG_BACKEND=chroma
export CHROMA_PERSIST_DIR=./chroma_db
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Check active retrieval backend:

- GET /rag/health  -> retrieval_backend field


Here's a clean, complete API Input Reference document for your frontend team:

---

# 🏦 MSME Loan Risk API — Input Reference Guide
**Endpoint:** `POST /predict`
**Content-Type:** `application/json`

---

## 📋 Complete Input Fields

### 👤 Section 1: Personal & Employment

| Field | Type | Required | Min | Max | Example | Description |
|---|---|---|---|---|---|---|
| `annual_income` | `number` | ✅ Yes | 1 | — | `45000` | Yearly income in ₹ |
| `emp_length` | `number` | ✅ Yes | 0 | 10 | `3` | Years at current job. Use `0` for less than 1 year, `10` for 10+ years |
| `application_type` | `string` | ✅ Yes | — | — | `"individual"` | See allowed values below |

**`application_type` allowed values:**
```
"individual"   → Single applicant
"joint app"    → Two applicants together
```

---

### 💰 Section 2: Loan Details

| Field | Type | Required | Min | Max | Example | Description |
|---|---|---|---|---|---|---|
| `loan_amount` | `number` | ✅ Yes | 1 | — | `15000` | Principal loan amount in ₹ |
| `term` | `number` | ✅ Yes | — | — | `36` | Loan duration. Only `36` or `60` months allowed |
| `int_rate` | `number` | ✅ Yes | 0 | 100 | `14.5` | Interest rate in **%** (e.g. 14.5 means 14.5%) |
| `installment` | `number` | ✅ Yes | 1 | — | `350.00` | Monthly EMI amount in ₹ |
| `purpose` | `string` | ✅ Yes | — | — | `"car"` | Reason for loan. See allowed values below |

**`term` allowed values:**
```
36   → 36 months (3 years)
60   → 60 months (5 years)
```

**`purpose` allowed values:**
```
"car"                  → Vehicle purchase
"credit_card"          → Credit card payoff
"debt_consolidation"   → Consolidating existing debts
"educational"          → Education expenses
"home_improvement"     → Home renovation
"house"                → House purchase
"major_purchase"       → Large item purchase
"medical"              → Medical expenses
"moving"               → Relocation expenses
"other"                → Any other reason
"renewable_energy"     → Solar/green energy
"small_business"       → Business expenses
"vacation"             → Travel
"wedding"              → Wedding expenses
```

---

### 📊 Section 3: Credit Profile

| Field | Type | Required | Allowed Values | Example | Description |
|---|---|---|---|---|---|
| `grade` | `string` | ✅ Yes | `a` `b` `c` `d` `e` `f` `g` | `"c"` | Bank-assigned loan grade. `a` = best, `g` = worst |
| `sub_grade` | `string` | ✅ Yes | `a1`–`g5` | `"c4"` | Sub-grade within grade. e.g. `a1, a2, a3, a4, a5, b1...g5` |
| `home_ownership` | `string` | ✅ Yes | See below | `"rent"` | Applicant's housing status |
| `verification_status` | `string` | ✅ Yes | See below | `"source verified"` | Document verification level |

**`home_ownership` allowed values:**
```
"own"       → Owns the house outright
"mortgage"  → House under mortgage/loan
"rent"      → Renting a house
"other"     → Other arrangement
```

**`verification_status` allowed values:**
```
"verified"         → All documents fully verified
"source verified"  → Source of income verified
"not verified"     → Documents not verified
```

**`sub_grade` full list:**
```
a1, a2, a3, a4, a5
b1, b2, b3, b4, b5
c1, c2, c3, c4, c5
d1, d2, d3, d4, d5
e1, e2, e3, e4, e5
f1, f2, f3, f4, f5
g1, g2, g3, g4, g5
```

---

### 📈 Section 4: Financial Details

| Field | Type | Required | Min | Example | Description |
|---|---|---|---|---|---|
| `dti` | `number` | ✅ Yes | 0 | `18.5` | Debt-to-Income ratio in **%**. e.g. `18.5` means 18.5% |
| `total_acc` | `integer` | ✅ Yes | 0 | `8` | Total number of credit accounts the applicant has |
| `total_payment` | `number` | ✅ Yes | 0 | `5000` | Total amount already paid towards this loan in ₹ |

---

### 📍 Section 5: Location

| Field | Type | Required | Example | Description |
|---|---|---|---|---|
| `address_state` | `string` | ✅ Yes | `"CA"` | 2-letter US state code (uppercase). e.g. `CA`, `GA`, `NY`, `TX` |

**Common `address_state` values:**
```
"CA" → California      "NY" → New York
"TX" → Texas           "FL" → Florida
"GA" → Georgia         "IL" → Illinois
"NJ" → New Jersey      "PA" → Pennsylvania
"OH" → Ohio            "VA" → Virginia
```

---

### 📅 Section 6: Date-Derived Fields
> These can be calculated by frontend from the loan dates before sending.

| Field | Type | Required | Default | Example | Description |
|---|---|---|---|---|---|
| `loan_age_months` | `number` | ⚪ Optional | `0` | `12` | How many months since loan was issued |
| `issue_month` | `integer` | ⚪ Optional | `1` | `3` | Month loan was issued (1=Jan, 12=Dec) |
| `issue_year` | `integer` | ⚪ Optional | `2021` | `2021` | Year loan was issued |
| `days_since_last_payment` | `number` | ⚪ Optional | `30` | `90` | Number of days since last payment was made |
| `days_since_credit_pull` | `number` | ⚪ Optional | `30` | `30` | Number of days since credit history was last checked |
| `days_to_next_payment` | `number` | ⚪ Optional | `15` | `15` | Number of days until next EMI is due |

**How frontend can calculate these:**
```javascript
// JavaScript helper
const today = new Date();

// loan_age_months
const issueDate = new Date("2021-02-11");
const loan_age_months = Math.floor((today - issueDate) / (1000 * 60 * 60 * 24 * 30));

// days_since_last_payment
const lastPayment = new Date("2021-04-13");
const days_since_last_payment = Math.floor((today - lastPayment) / (1000 * 60 * 60 * 24));

// days_to_next_payment
const nextPayment = new Date("2021-05-13");
const days_to_next_payment = Math.floor((nextPayment - today) / (1000 * 60 * 60 * 24));

// issue_month and issue_year
const issue_month = issueDate.getMonth() + 1;  // +1 because JS months are 0-indexed
const issue_year  = issueDate.getFullYear();
```

---

## ✅ Complete Sample Request Body

``````json
{
    "annual_income"           : 45000,
    "emp_length"              : 3,
    "application_type"        : "individual",
    "loan_amount"             : 15000,
    "term"                    : 36,
    "int_rate"                : 14.5,
    "installment"             : 350.00,
    "purpose"                 : "car",
    "grade"                   : "c",
    "sub_grade"               : "c4",
    "home_ownership"          : "rent",
    "verification_status"     : "source verified",
    "dti"                     : 18.5,
    "total_acc"               : 8,
    "total_payment"           : 5000,
    "address_state"           : "CA",
    "loan_age_months"         : 12,
    "issue_month"             : 3,
    "issue_year"              : 2021,
    "days_since_last_payment" : 90,
    "days_since_credit_pull"  : 30,
    "days_to_next_payment"    : 15
}
```

---

## ❌ Common Validation Errors

| Error | Cause | Fix |
|---|---|---|
| `annual_income must be > 0` | Sent `0` or negative | Send actual income value |
| `term must be 36 or 60` | Sent `48` or other value | Only `36` or `60` allowed |
| `int_rate must be between 0 and 100` | Sent `0.145` (decimal) | Send `14.5` not `0.145` |
| `dti must be >= 0` | Sent negative value | Send `18.5` not `-18.5` |
| `invalid sub_grade` | Sent `"C4"` uppercase | Always send **lowercase** `"c4"` |
| `invalid grade` | Sent `"C"` uppercase | Always send **lowercase** `"c"` |
| `invalid application_type` | Sent `"INDIVIDUAL"` | Always send **lowercase** `"individual"` |
| `invalid home_ownership` | Sent `"RENT"` | Always send **lowercase** `"rent"` |
| `invalid verification_status` | Sent `"Verified"` | Always send **lowercase** `"verified"` |

---

## 📤 What API Returns (Response Summary)

```json
{
    "prediction"          : 0,
    "decision"            : "APPROVE",
    "risk_band"           : "GREEN",
    "default_probability" : 0.2134,
    "confidence"          : 78.66,
    "health_card": {
        "liquidity"   : 72,
        "solvency"    : 65,
        "growth"      : 45,
        "compliance"  : 78,
        "repayment"   : 80,
        "overall"     : 714
    },
    "explanation": {
        "top_risk_factors": {
            "dti"      : 0.0823,
            "int_rate" : 0.0512
        },
        "top_safety_factors": {
            "grade"         : -0.0921,
            "annual_income" : -0.0756
        },
        "base_value" : 0.1823
    },
    "recommended_amount" : 15000,
    "recommended_rate"   : 14.5,
    "recommended_term"   : 36,
    "model_name"         : "XGBoost",
    "model_auc"          : 92.4,
    "request_id"         : "f3a2b1c4-d5e6-7890-abcd-ef1234567890",
    "timestamp"          : "2026-07-08T13:37:00Z"
}
```

### Response Field Meanings:

| Field | Type | Meaning |
|---|---|---|
| `prediction` | `0` or `1` | `0` = Good Loan, `1` = Bad Loan |
| `decision` | `string` | `APPROVE` / `MANUAL REVIEW` / `REJECT` |
| `risk_band` | `string` | `GREEN` / `AMBER` / `RED` |
| `default_probability` | `number` | Probability of default (0.0 to 1.0) |
| `confidence` | `number` | Model confidence in % (0 to 100) |
| `health_card.liquidity` | `number` | Cash flow health score (0–100) |
| `health_card.solvency` | `number` | Debt burden score (0–100) |
| `health_card.growth` | `number` | Income growth score (0–100) |
| `health_card.compliance` | `number` | Document & grade score (0–100) |
| `health_card.repayment` | `number` | Payment behaviour score (0–100) |
| `health_card.overall` | `number` | Combined score (0–1000) |
| `explanation.top_risk_factors` | `object` | Features increasing default risk |
| `explanation.top_safety_factors` | `object` | Features decreasing default risk |
| `recommended_amount` | `number` | Loan amount bank recommends (₹) |
| `recommended_rate` | `number` | Interest rate bank recommends (%) |
| `recommended_term` | `number` | Tenure bank recommends (months) |
| `request_id` | `string` | Unique ID for audit trail |
| `timestamp` | `string` | UTC time of prediction |

---

## 🚦 Decision Logic (For Frontend Display)

```
default_probability < 0.35  →  decision = "APPROVE"       risk_band = "GREEN" 🟢
default_probability < 0.65  →  decision = "MANUAL REVIEW" risk_band = "AMBER" 🟡
default_probability >= 0.65 →  decision = "REJECT"        risk_band = "RED"   🔴
```

---

## 📐 Field Summary Card (Quick Reference)

```
┌─────────────────────────────────────────────────────────────┐
│              REQUIRED FIELDS (15 total)                     │
├──────────────────────┬──────────┬───────────────────────────┤
│ Field                │ Type     │ Example                   │
├──────────────────────┼──────────┼───────────────────────────┤
│ annual_income        │ number   │ 45000                     │
│ emp_length           │ number   │ 3  (0–10)                 │
│ application_type     │ string   │ "individual"              │
│ loan_amount          │ number   │ 15000                     │
│ term                 │ number   │ 36 or 60                  │
│ int_rate             │ number   │ 14.5  (percent)           │
│ installment          │ number   │ 350.00                    │
│ purpose              │ string   │ "car"                     │
│ grade                │ string   │ "c"  (a–g lowercase)      │
│ sub_grade            │ string   │ "c4" (a1–g5 lowercase)    │
│ home_ownership       │ string   │ "rent"                    │
│ verification_status  │ string   │ "source verified"         │
│ dti                  │ number   │ 18.5  (percent)           │
│ total_acc            │ integer  │ 8                         │
│ total_payment        │ number   │ 5000                      │
│ address_state        │ string   │ "CA"  (uppercase)         │
├──────────────────────┴──────────┴───────────────────────────┤
│              OPTIONAL FIELDS (6 total)                      │
├──────────────────────┬──────────┬───────────────────────────┤
│ loan_age_```
│ loan_age_months          │ number   │ 12  (default: 0)          │
│ issue_month              │ integer  │ 3   (1–12, default: 1)    │
│ issue_year               │ integer  │ 2021 (default: 2021)      │
│ days_since_last_payment  │ number   │ 90  (default: 30)         │
│ days_since_credit_pull   │ number   │ 30  (default: 30)         │
│ days_to_next_payment     │ number   │ 15  (default: 15)         │
└──────────────────────────┴──────────┴───────────────────────────┘
```

---

## 🌐 All API Endpoints (Frontend Reference)

| Method | Endpoint | Purpose | Input |
|---|---|---|---|
| `GET` | `/health` | Check if API is running | None |
| `GET` | `/model/info` | Get model performance stats | None |
| `POST` | `/predict` | Single loan prediction | Full input JSON |
| `POST` | `/predict/batch` | Multiple loans at once | `{"applications": [...]}` |
| `POST` | `/predict/summary` | Portfolio risk stats | `{"applications": [...]}` |
| `POST` | `/health-card` | Health card score only | Full input JSON |
| `POST` | `/explain` | SHAP explanation only | Full input JSON |

---

## 📦 Batch Request Format

> Use this when submitting **multiple applications** at once (max 100)

```json
{
    "applications": [
        {
            "annual_income"          : 45000,
            "emp_length"             : 3,
            "application_type"       : "individual",
            "loan_amount"            : 15000,
            "term"                   : 36,
            "int_rate"               : 14.5,
            "installment"            : 350.00,
            "purpose"                : "car",
            "grade"                  : "c",
            "sub_grade"              : "c4",
            "home_ownership"         : "rent",
            "verification_status"    : "source verified",
            "dti"                    : 18.5,
            "total_acc"              : 8,
            "total_payment"          : 5000,
            "address_state"          : "CA"
        },
        {
            "annual_income"          : 80000,
            "emp_length"             : 7,
            "application_type"       : "individual",
            "loan_amount"            : 25000,
            "term"                   : 60,
            "int_rate"               : 9.5,
            "installment"            : 520.00,
            "purpose"                : "debt_consolidation",
            "grade"                  : "a",
            "sub_grade"              : "a2",
            "home_ownership"         : "mortgage",
            "verification_status"    : "verified",
            "dti"                    : 8.2,
            "total_acc"              : 15,
            "total_payment"          : 12000,
            "address_state"          : "NY"
        }
    ]
}
```

---

## ⚙️ Frontend Integration Checklist

```
✅ All string fields must be LOWERCASE
   except address_state which must be UPPERCASE (e.g. "CA")

✅ int_rate and dti must be in PERCENTAGE format
   Send 14.5 (not 0.145)

✅ term must be EXACTLY 36 or 60
   No other values accepted

✅ grade must be single letter: a, b, c, d, e, f, g

✅ sub_grade must be letter + number: a1, b3, g5 etc.

✅ Optional date fields have defaults
   Safe to omit if not available

✅ annual_income and loan_amount must be POSITIVE numbers

✅ total_payment can be 0 for new loans
   (no payments made yet)

✅ emp_length must be between 0 and 10
   0  = less than 1 year
   10 = 10 or more years
```

---

## 🔗 HTTP Headers Required

```
Content-Type  : application/json
Accept        : application/json
```

---

## 📊 Three Different Loan Profiles (Test Cases for Frontend)

### 🟢 Low Risk Applicant (Expected: APPROVE)
```json
{
    "annual_income"       : 120000,
    "emp_length"          : 10,
    "application_type"    : "individual",
    "loan_amount"         : 10000,
    "term"                : 36,
    "int_rate"            : 7.5,
    "installment"         : 310.00,
    "purpose"             : "debt_consolidation",
    "grade"               : "a",
    "sub_grade"           : "a1",
    "home_ownership"      : "own",
    "verification_status" : "verified",
    "dti"                 : 5.2,
    "total_acc"           : 20,
    "total_payment"       : 8000,
    "address_state"       : "CA"
}
```

### 🟡 Medium Risk Applicant (Expected: MANUAL REVIEW)
```json
{
    "annual_income"       : 45000,
    "emp_length"          : 3,
    "application_type"    : "individual",
    "loan_amount"         : 15000,
    "term"                : 36,
    "int_rate"            : 14.5,
    "installment"         : 350.00,
    "purpose"             : "car",
    "grade"               : "c",
    "sub_grade"           : "c4",
    "home_ownership"      : "rent",
    "verification_status" : "source verified",
    "dti"                 : 18.5,
    "total_acc"           : 8,
    "total_payment"       : 5000,
    "address_state"       : "GA"
}
```

### 🔴 High Risk Applicant (Expected: REJECT)
```json
{
    "annual_income"       : 18000,
    "emp_length"          : 0,
    "application_type"    : "individual",
    "loan_amount"         : 30000,
    "term"                : 60,
    "int_rate"            : 24.5,
    "installment"         : 890.00,
    "purpose"             : "small_business",
    "grade"               : "g",
    "sub_grade"           : "g5",
    "home_ownership"      : "rent",
    "verification_status" : "not verified",
    "dti"                 : 48.5,
    "total_acc"           : 3,
    "total_payment"       : 0,
    "address_state"       : "TX"
}
```

---

## 🗂️ Complete Field Reference (One Page Summary)

```
┌────────────────────────────────────────────────────────────────────┐
│                    MSME LOAN API — FIELD REFERENCE                 │
├─────────────────────┬────────┬──────────┬────────────────────────  │
│ FIELD               │ TYPE   │ REQUIRED │ VALID VALUES             │
├─────────────────────┼────────┼──────────┼──────────────────────────┤
│ annual_income       │ float  │ YES      │ > 0                      │
│ emp_length          │ float  │ YES      │ 0 to 10                  │
│ application_type    │ string │ YES      │ "individual","joint app" │
│ loan_amount         │ float  │ YES      │ > 0                      │
│ term                │ int    │ YES      │ 36 or 60                ```
│ int_rate            │ float  │ YES      │ 0.1 to 99.9 (%)          │
│ installment         │ float  │ YES      │ > 0                      │
│ purpose             │ string │ YES      │ "car","credit_card",     │
│                     │        │          │ "debt_consolidation",    │
│                     │        │          │ "educational","house",   │
│                     │        │          │ "home_improvement",      │
│                     │        │          │ "major_purchase",        │
│                     │        │          │ "medical","moving",      │
│                     │        │          │ "other","renewable_      │
│                     │        │          │ energy","small_business",│
│                     │        │          │ "vacation","wedding"     │
│ grade               │ string │ YES      │ "a","b","c","d",         │
│                     │        │          │ "e","f","g"              │
│ sub_grade           │ string │ YES      │ "a1" to "g5"             │
│ home_ownership      │ string │ YES      │ "own","mortgage",        │
│                     │        │          │ "rent","other"           │
│ verification_status │ string │ YES      │ "verified",              │
│                     │        │          │ "source verified",       │
│                     │        │          │ "not verified"           │
│ dti                 │ float  │ YES      │ >= 0 (%)                 │
│ total_acc           │ int    │ YES      │ >= 0                     │
│ total_payment       │ float  │ YES      │ >= 0                     │
│ address_state       │ string │ YES      │ 2-letter UPPERCASE code  │
│                     │        │          │ e.g. "CA","NY","TX"      │
├─────────────────────┼────────┼──────────┼──────────────────────────┤
│ loan_age_months     │ float  │ NO       │ >= 0  (default: 0)       │
│ issue_month         │ int    │ NO       │ 1–12  (default: 1)       │
│ issue_year          │ int    │ NO       │ 4-digit (default: 2021)  │
│ days_since_last_    │ float  │ NO       │ >= 0  (default: 30)      │
│   payment           │        │          │                          │
│ days_since_credit_  │ float  │ NO       │ >= 0  (default: 30)      │
│   pull              │        │          │                          │
│ days_to_next_       │ float  │ NO       │ any   (default: 15)      │
│   payment           │        │          │ negative = overdue       │
└─────────────────────┴────────┴──────────┴──────────────────────────┘
```

---

## ⚠️ Important Notes for Frontend Team

```
1. CASE SENSITIVITY
   ─────────────────
   All string values must be LOWERCASE except address_state
   ✅ "individual"   ❌ "Individual" or "INDIVIDUAL"
   ✅ "verified"     ❌ "Verified"   or "VERIFIED"
   ✅ "CA"           ❌ "ca"         or "Ca"

2. PERCENTAGE FIELDS
   ──────────────────
   int_rate and dti are in PERCENTAGE not decimal
   ✅ int_rate: 14.5   means 14.5%
   ❌ int_rate: 0.145  is WRONG

3. NEW LOAN (no payments yet)
   ───────────────────────────
   If applicant is applying for first time:
   total_payment           = 0
   days_since_last_payment = 0
   loan_age_months         = 0

4. emp_length MAPPING
   ───────────────────
   Less than 1 year  → send 0
   1 year            → send 1
   2 years           → send 2
   ...
   10 or more years  → send 10

5. days_to_next_payment
   ─────────────────────
   Positive = payment due in future   e.g. 15
   Zero     = payment due today       e.g. 0
   Negative = payment is overdue      e.g. -5

6. BATCH LIMIT
   ────────────
   Maximum 100 applications per batch request
   For more, split into multiple requests

7. RESPONSE TIME
   ──────────────
   Single prediction  → ~200–500ms
   Batch (100 apps)   → ~5–15 seconds
```

---

## 🧪 Quick Test Using JavaScript (Fetch)

```javascript
// Single Prediction
const response = await fetch('http://localhost:8000/predict', {
    method  : 'POST',
    headers : {
        'Content-Type' : 'application/json',
        'Accept'       : 'application/json'
    },
    body: JSON.stringify({
        annual_income        : 45000,
        emp_length           : 3,
        application_type     : "individual",
        loan_amount          : 15000,
        term                 : 36,
        int_rate             : 14.5,
        installment          : 350.00,
        purpose              : "car",
        grade                : "c",
        sub_grade            : "c4",
        home_ownership       : "rent",
        verification_status  : "source verified",
        dti                  : 18.5,
        total_acc            : 8,
        total_payment        : 5000,
        address_state        : "CA",
        loan_age_months      : 12,
        issue_month          : 3,
        issue_year           : 2021,
        days_since_last_payment : 90,
        days_since_credit_pull  : 30,
        days_to_next_payment    : 15
    })
});

const result = await response.json();

// Display decision
console.log('Decision     :', result.decision);
console.log('Risk Band    :', result.risk_band);
console.log('Default Prob :', result.default_probability * 100 + '%');
console.log('Health Score :', result.health_card.overall + '/1000');
```

---

## 🧪 Quick Test Using Axios (React/Vue)

```javascript
import axios from 'axios';

const predictLoan = async (formData) => {
    try {
        const { data } = await axios.post(
            'http://localhost:8000/predict',
            {
                annual_income           : formData.annualIncome,
                emp_length              : formData.empLength,
                application_type        : formData.applicationType.toLowerCase(),
                loan_amount             : formData.loanAmount,
                term                    : formData.term,
                int_rate                : formData.intRate,
                installment             : formData.installment,
                purpose                 : formData.purpose.toLowerCase(),
                grade                   : formData.grade.toLowerCase(),
                sub_grade               : formData.subGrade.toLowerCase(),
                home_ownership          : formData.homeOwnership.toLowerCase(),
                verification_status     : formData.verificationStatus.toLowerCase(),
                dti                     : formData.dti,
                total_acc               : formData.totalAcc,
                total_payment           : formData.totalPayment,
                address_state           : formData.addressState.toUpperCase(),
                loan_age_months         : formData.loanAgeMonths         || 0,
                issue_month             : formData.issueMonth             || 1,
                issue_year              : formData.issueYear              || 2021,
                days_since_last_payment : formData.daysSinceLastPayment   || 30,
                days_since_```javascript
                days_since_credit_pull  : formData.daysSinceCreditPull   || 30,
                days_to_next_payment    : formData.daysToNextPayment      || 15
            },
            {
                headers: {
                    'Content-Type' : 'application/json',
                    'Accept'       : 'application/json'
                }
            }
        );

        return {
            decision            : data.decision,
            riskBand            : data.risk_band,
            defaultProbability  : data.default_probability,
            confidence          : data.confidence,
            healthCard          : data.health_card,
            explanation         : data.explanation,
            recommendedAmount   : data.recommended_amount,
            recommendedRate     : data.recommended_rate,
            recommendedTerm     : data.recommended_term,
            requestId           : data.request_id,
            timestamp           : data.timestamp
        };

    } catch (error) {
        if (error.response?.status === 422) {
            // Validation error — show field-level errors to user
            console.error('Validation Error:', error.response.data.detail);
            throw new Error('Invalid input. Please check all fields.');
        }
        throw new Error('Prediction failed. Please try again.');
    }
};
```

---

## 📱 Suggested Frontend Form Fields (UI Mapping)

```
┌─────────────────────────────────────────────────────────────┐
│              LOAN APPLICATION FORM — UI FIELDS              │
├──────────────────────┬──────────────────────────────────────┤
│ UI Label             │ API Field                            │
├──────────────────────┼──────────────────────────────────────┤
│ Annual Income (₹)    │ annual_income                        │
│ Years Employed       │ emp_length (dropdown 0–10)           │
│ Application Type     │ application_type (dropdown)          │
│ Loan Amount (₹)      │ loan_amount                          │
│ Loan Tenure          │ term (radio: 36 or 60 months)        │
│ Interest Rate (%)    │ int_rate                             │
│ Monthly EMI (₹)      │ installment                          │
│ Loan Purpose         │ purpose (dropdown)                   │
│ Loan Grade           │ grade (dropdown a–g)                 │
│ Loan Sub-Grade       │ sub_grade (dropdown a1–g5)           │
│ Home Ownership       │ home_ownership (dropdown)            │
│ Verification Status  │ verification_status (dropdown)       │
│ Debt-to-Income (%)   │ dti                                  │
│ Total Accounts       │ total_acc                            │
│ Total Paid So Far(₹) │ total_payment                        │
│ State                │ address_state (dropdown)             │
│ Loan Issue Date      │ → compute loan_age_months,           │
│                      │   issue_month, issue_year            │
│ Last Payment Date    │ → compute days_since_last_payment    │
│ Last Credit Check    │ → compute days_since_credit_pull     │
│ Next Payment Date    │ → compute days_to_next_payment       │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 🎨 Suggested Frontend Display for Response

```
┌─────────────────────────────────────────────────────────────┐
│                  LOAN ASSESSMENT RESULT                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Decision    : ✅ APPROVE          Risk : 🟢 GREEN         │
│   Default Risk: 21.34%              Confidence: 78.66%      │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  FINANCIAL HEALTH CARD                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   💧 Liquidity   [████████░░]  72/100                       │
│   🏗️  Solvency   [██████░░░░]  65/100                       │
│   📈 Growth      [████░░░░░░]  45/100                       │
│   ✅ Compliance  [███████░░░]  78/100                       │
│   🔄 Repayment   [████████░░]  80/100                       │
│                                                             │
│   🏆 Overall Score  :  714 / 1000                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  LOAN OFFER                                 │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Amount  : ₹ 15,000                                        │
│   Rate    : 14.5%                                           │
│   Tenure  : 36 months                                       │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                  WHY THIS DECISION?                         │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   ⚠️  Risk Factors                                          │
│   → High DTI ratio          (+0.0823)                       │
│   → Interest rate is high   (+0.0512)                       │
│                                                             │
│   ✅ Positive Factors                                       │
│   → Good loan grade         (-0.0921)                       │
│   → Strong annual income    (-0.0756)                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

This is the **complete API Input Reference** your frontend team needs. Share this document directly — it covers every field, allowed value, validation rule, sample requests, response format, JavaScript integration code, and UI mapping. Would you like me to convert this into a proper **Postman Collection JSON** or a **Swagger YAML** file that the frontend team can import directly?