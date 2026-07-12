# ============================================================
# main.py
# FastAPI Application — MSME Loan Risk Assessment API 
# ============================================================

import uuid
import time
import joblib
import shap
import numpy   as np
import pandas  as pd
from pathlib import Path
from tempfile import NamedTemporaryFile

from datetime  import datetime
from fastapi   import FastAPI, HTTPException, Request, UploadFile, File, status
from fastapi.middleware.cors        import CORSMiddleware
from fastapi.middleware.gzip        import GZipMiddleware
from fastapi.responses              import JSONResponse, FileResponse
from fastapi.staticfiles            import StaticFiles
from contextlib                     import asynccontextmanager

from schemas.loan_schema  import (
    LoanApplicationRequest,
    LoanPredictionResponse,
    BatchLoanRequest,
    BatchLoanResponse,
    BatchPredictionResult,
    HealthCheckResponse,
    HealthCardScores,
    ShapExplanation
)
from services.prediction  import encode_application, run_shap_explanation
from services.prediction  import GRADE_MAP, VERIFY_MAP
from utils.health_card    import (
    compute_health_card,
    get_risk_band,
    get_decision,
    compute_loan_offer
)
from graph.workflow import run_assessment_workflow, bootstrap_policy_index, get_retrieval_backend

# ============================================================
# GLOBAL MODEL STORE
# ============================================================

model_store = {
    "model"    : None,
    "scaler"   : None,
    "metadata" : None,
    "explainer": None,
}

BASE_DIR = Path(__file__).resolve().parent
FRONTEND_DIR = BASE_DIR / "frontend"
UPLOADS_DIR = BASE_DIR / "uploads"
POLICIES_DIR = BASE_DIR / "knowledge_base" / "policies"

UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
POLICIES_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# LIFESPAN — Load Model on Startup
# ============================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load all model artifacts on startup, release on shutdown."""

    print("🚀 Starting MSME Loan Risk API...")
    print("🔄 Loading model artifacts...")

    try:
        model_store["model"]    = joblib.load("model/best_model.pkl")
        model_store["scaler"]   = joblib.load("model/scaler.pkl")
        model_store["metadata"] = joblib.load("model/model_metadata.pkl")

        # Build SHAP explainer once at startup (faster per-request)
        model_store["explainer"] = shap.TreeExplainer(model_store["model"])

        meta = model_store["metadata"]
        print(f"✅ Model loaded     : {meta['model_name']}")
        print(f"✅ AUC-ROC          : {meta['auc_roc']}%")
        print(f"✅ Total features   : {meta['n_features']}")
        print(f"✅ API ready        : http://localhost:8000")
        print(f"✅ Swagger docs     : http://localhost:8000/docs")

        # Prime policy retrieval index for RAG assessment flow.
        indexed_count = bootstrap_policy_index(POLICIES_DIR)
        print(f"✅ Policy chunks    : {indexed_count}")

    except FileNotFoundError as e:
        print(f"❌ Model file not found: {e}")
        print("   Make sure best_model.pkl, scaler.pkl, model_metadata.pkl")
        print("   are inside the model/ directory.")
        raise

    yield   # API runs here

    # Shutdown cleanup
    print("🛑 Shutting down API — releasing resources...")
    model_store.clear()


# ============================================================
# FASTAPI APP INITIALIZATION
# ============================================================

app = FastAPI(
    title       = "🏦 MSME Loan Risk Assessment API",
    description = """
## AI/ML-Powered MSME Financial Health Card & Credit Decision API

### Features
- ✅ **Instant Loan Decision** — Approve / Manual Review / Reject
- 📊 **Financial Health Card** — 5-dimension score (Liquidity, Solvency, Growth, Compliance, Repayment)
- 🔍 **SHAP Explainability** — Understand exactly why a loan was approved or rejected
- 📦 **Batch Processing** — Score up to 100 applications in one request
- 🔒 **RBI Compliant** — Explainable AI decisions with audit trail

### Risk Bands
| Band  | Probability | Decision       |
|-------|-------------|----------------|
| 🟢 GREEN | < 35%   | Auto Approve   |
| 🟡 AMBER | 35–65%  | Manual Review  |
| 🔴 RED   | > 65%   | Auto Reject    |
    """,
    version     = "1.0.0",
    lifespan    = lifespan,
    docs_url    = "/docs",
    redoc_url   = "/redoc"
)

if FRONTEND_DIR.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIR)), name="assets")

# ============================================================
# MIDDLEWARE
# ============================================================

# CORS — allow all origins (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins     = ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)

# GZip compression for large responses
app.add_middleware(GZipMiddleware, minimum_size=1000)


# ============================================================
# REQUEST LOGGING MIDDLEWARE
# ============================================================

@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response   = await call_next(request)
    duration   = round((time.time() - start_time) * 1000, 2)
    print(f"📥 {request.method} {request.url.path} → {response.status_code} ({duration}ms)")
    response.headers["X-Process-Time-Ms"] = str(duration)
    return response


# ============================================================
# HELPER — Core Prediction Logic
# ============================================================

def _predict_single(app_request: LoanApplicationRequest) -> dict:
    """
    Internal helper: encode → scale → predict → explain → health card.
    Returns a dict with all prediction components.
    """
    model    = model_store["model"]
    scaler   = model_store["scaler"]
    meta     = model_store["metadata"]
    explainer= model_store["explainer"]
    features = meta["features"]

    # 1. Encode application
    encoded_df, raw_encoded = encode_application(app_request, features)

    # 2. Scale features
    scaled_array = scaler.transform(encoded_df)
    scaled_df    = pd.DataFrame(scaled_array, columns=features)

    # 3. Predict
    #"What is the probability that this borrower will DEFAULT (fail to repay)?"
    prediction   = int(model.predict(scaled_array)[0])
    probabilities= model.predict_proba(scaled_array)[0]
    default_prob = round(float(probabilities[1]), 4)
    confidence   = round(float(max(probabilities)) * 100, 2)

    # 4. Decision & Risk Band
    decision     = get_decision(default_prob)
    risk_band    = get_risk_band(default_prob)

    # 5. SHAP Explanation
    shap_result  = run_shap_explanation(explainer, scaled_df, features)

    # 6. Financial Health Card
    health_input = {
        **raw_encoded,
        'dti'                    : app_request.dti,
        'loan_to_income_ratio'   : raw_encoded['loan_to_income_ratio'],
        'emi_to_income_ratio'    : raw_encoded['emi_to_income_ratio'],
        'payment_ratio'          : raw_encoded['payment_ratio'],
        'days_since_last_payment': app_request.days_since_last_payment,
    }
    health_card = compute_health_card(health_input)

    # 7. Loan Offer (only if not rejected)
    loan_offer = {}
    if decision != "REJECT":
        loan_offer = compute_loan_offer(
            requested_amount = app_request.loan_amount,
            requested_rate   = app_request.int_rate,
            requested_term   = int(app_request.term),
            probability      = default_prob,
            health_score     = health_card.overall
        )

    return {
        "prediction"          : prediction,
        "decision"            : decision,
        "risk_band"           : risk_band,
        "default_probability" : default_prob,
        "confidence"          : confidence,
        "health_card"         : health_card,
        "shap_result"         : shap_result,
        "loan_offer"          : loan_offer,
        "model_name"          : meta["model_name"],
        "model_auc"           : meta["auc_roc"],
    }





# ============================================================
# ROUTE 1: Health Check
# ============================================================

@app.get(
    "/health",
    response_model = HealthCheckResponse,
    tags           = ["System"],
    summary        = "API Health Check",
    description    = "Check if the API and model are loaded and ready."
)
async def health_check():
    meta = model_store.get("metadata", {})
    return HealthCheckResponse(
        status         = "healthy" if model_store["model"] else "unhealthy",
        model_loaded   = model_store["model"]    is not None,
        scaler_loaded  = model_store["scaler"]   is not None,
        model_name     = meta.get("model_name",  "unknown"),
        model_auc      = meta.get("auc_roc",     0.0),
        total_features = meta.get("n_features",  0),
        api_version    = "1.0.0"
    )


# ============================================================
# ROUTE 2: Single Loan Prediction
# ============================================================

@app.post(
    "/predict",
    response_model = LoanPredictionResponse,
    tags           = ["Prediction"],
    summary        = "Single Loan Application Assessment",
    description    = """
Submit a single MSME loan application and receive:
- **Instant decision** (Approve / Manual Review / Reject)
- **Risk band** (Green / Amber / Red)
- **Default probability** (0–100%)
- **Financial Health Card** (5-dimension score)
- **SHAP explanation** (top risk & safety factors)
- **Recommended loan offer** (if approved)
    """,
    status_code    = status.HTTP_200_OK
)
async def predict_loan(application: LoanApplicationRequest):

    # Guard: ensure model is loaded
    if not model_store["model"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model not loaded. Please try again shortly."
        )

    try:
        result     = _predict_single(application)
        request_id = str(uuid.uuid4())
        timestamp  = datetime.utcnow().isoformat() + "Z"

        return LoanPredictionResponse(
            prediction          = result["prediction"],
            decision            = result["decision"],
            risk_band           = result["risk_band"],
            default_probability = result["default_probability"],
            confidence          = result["confidence"],
            health_card         = result["health_card"],
            explanation         = ShapExplanation(
                top_risk_factors   = result["shap_result"]["top_risk_factors"],
                top_safety_factors = result["shap_result"]["top_safety_factors"],
                base_value         = result["shap_result"]["base_value"]
            ),
            recommended_amount  = result["loan_offer"].get("recommended_amount"),
            recommended_rate    = result["loan_offer"].get("recommended_rate"),
            recommended_term    = result["loan_offer"].get("recommended_term"),
            model_name          = result["model_name"],
            model_auc           = result["model_auc"],
            request_id          = request_id,
            timestamp           = timestamp
        )

    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Prediction failed: {str(e)}"
        )


# ============================================================
# ROUTE 3: Batch Prediction
# ============================================================


@app.post(
    "/predict/batch",
    response_model = BatchLoanResponse,
    tags           = ["Prediction"],
    summary        = "Batch Loan Applications Assessment",
    description    = "Submit up to 100 loan applications at once and get decisions for all.",
    status_code    = status.HTTP_200_OK
)
async def predict_batch(batch: BatchLoanRequest):

    if not model_store["model"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model not loaded."
        )

    if len(batch.applications) == 0:
        raise HTTPException(
            status_code = status.HTTP_400_BAD_REQUEST,
            detail      = "No applications provided."
        )

    start_time  = time.time()
    results     = []
    approved    = 0
    rejected    = 0
    manual      = 0

    for idx, app_request in enumerate(batch.applications):
        try:
            result = _predict_single(app_request)

            if result["decision"] == "APPROVE":
                approved += 1
            elif result["decision"] == "REJECT":
                rejected += 1
            else:
                manual   += 1

            results.append(BatchPredictionResult(
                index                = idx,
                prediction           = result["prediction"],
                decision             = result["decision"],
                risk_band            = result["risk_band"],
                default_probability  = result["default_probability"],
                overall_health_score = result["health_card"].overall
            ))

        except Exception as e:
            # Don't fail entire batch — log and continue
            print(f"⚠️  Batch item {idx} failed: {str(e)}")
            results.append(BatchPredictionResult(
                index                = idx,
                prediction           = -1,
                decision             = "ERROR",
                risk_band            = "RED",
                default_probability  = 1.0,
                overall_health_score = 0
            ))
            rejected += 1

    processing_time = round((time.time() - start_time) * 1000, 2)

    return BatchLoanResponse(
        total_applications = len(batch.applications),
        approved           = approved,
        rejected           = rejected,
        manual_review      = manual,
        results            = results,
        processing_time_ms = processing_time
    )


# ============================================================
# ROUTE 4: Financial Health Card Only
# ============================================================

@app.post(
    "/health-card",
    response_model = HealthCardScores,
    tags           = ["Health Card"],
    summary        = "Get Financial Health Card Only",
    description    = "Compute the 5-dimension Financial Health Card without full prediction.",
    status_code    = status.HTTP_200_OK
)
async def get_health_card(application: LoanApplicationRequest):

    try:
        lti_ratio = round(application.loan_amount / application.annual_income, 4) \
                    if application.annual_income > 0 else 0
        pay_ratio = round(application.total_payment / application.loan_amount, 4) \
                    if application.loan_amount > 0 else 0
        emi_ratio = round(application.installment / (application.annual_income / 12), 4) \
                    if application.annual_income > 0 else 0

        health_input = {
            
            'grade_encoded'              : GRADE_MAP.get(application.grade.lower(), 3),
            'verification_status_encoded': VERIFY_MAP.get(application.verification_status.lower(), 0),
            'annual_income'              : application.annual_income,
            'dti'                        : application.dti,
            'loan_to_income_ratio'       : lti_ratio,
            'payment_ratio'              : pay_ratio,
            'emi_to_income_ratio'        : emi_ratio,
            'days_since_last_payment'    : application.days_since_last_payment,
        }
        return compute_health_card(health_input)

    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"Health card computation failed: {str(e)}"
        )


# ============================================================
# ROUTE 5: SHAP Explanation Only
# ============================================================

@app.post(
    "/explain",
    tags        = ["Explainability"],
    summary     = "Get SHAP Explanation for a Loan Application",
    description = "Returns top risk-increasing and risk-decreasing features for a given application.",
    status_code = status.HTTP_200_OK
)
async def explain_prediction(application: LoanApplicationRequest):

    if not model_store["model"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model not loaded."
        )

    try:
        meta      = model_store["metadata"]
        scaler    = model_store["scaler"]
        explainer = model_store["explainer"]
        features  = meta["features"]

        encoded_df, _ = encode_application(application, features)
        scaled_array  = scaler.transform(encoded_df)
        scaled_df     = pd.DataFrame(scaled_array, columns=features)

        shap_result   = run_shap_explanation(explainer, scaled_df, features, top_n=10)

        return {
            "request_id"         : str(uuid.uuid4()),
            "timestamp"          : datetime.utcnow().isoformat() + "Z",
            "top_risk_factors"   : shap_result["top_risk_factors"],
            "top_safety_factors" : shap_result["top_safety_factors"],
            "base_value"         : shap_result["base_value"],
            "interpretation"     : {
                "note"      : "Positive SHAP values increase default risk. Negative values decrease it.",
                "base_value": f"Average model prediction across training data: {shap_result['base_value']}"
            }
        }

    except Exception as e:
        raise HTTPException(
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail      = f"SHAP explanation failed: {str(e)}"
        )


# ============================================================
# ROUTE 6: Model Information
# ============================================================

@app.get(
    "/model/info",
    tags        = ["System"],
    summary     = "Get Model Metadata",
    description = "Returns information about the currently loaded ML model."
)
async def model_info():
    meta = model_store.get("metadata", {})
    if not meta:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model metadata not available."
        )
    return {
        "model_name"     : meta.get("model_name"),
        "auc_roc"        : meta.get("auc_roc"),
        "f1_score"       : meta.get("f1_score"),
        "accuracy"       : meta.get("accuracy"),
        "n_features"     : meta.get("n_features"),
        "train_samples"  : meta.get("train_samples"),
        "test_samples"   : meta.get("test_samples"),
        "api_version"    : "1.0.0",
        "framework"      : "FastAPI + XGBoost + SHAP",
        "last_updated"   : datetime.utcnow().isoformat() + "Z"
    }


# ============================================================
# ROUTE 7: Risk Band Summary Statistics
# ============================================================

@app.post(
    "/predict/summary",
    tags        = ["Prediction"],
    summary     = "Batch Summary Statistics",
    description = "Submit multiple applications and get portfolio-level risk summary.",
    status_code = status.HTTP_200_OK
)
async def predict_summary(batch: BatchLoanRequest):

    if not model_store["model"]:
        raise HTTPException(
            status_code = status.HTTP_503_SERVICE_UNAVAILABLE,
            detail      = "Model not loaded."
        )

    probabilities = []
    health_scores = []
    decisions     = []

    for app_request in batch.applications:
        try:
            result = _predict_single(app_request)
            probabilities.append(result["default_probability"])
            health_scores.append(result["health_card"].overall)
            decisions.append(result["decision"])
        except:
            probabilities.append(1.0)
            health_scores.append(0)
            decisions.append("ERROR")

    prob_array   = np.array(probabilities)
    health_array = np.array(health_scores)

    return {
        "request_id"              : str(uuid.uuid4()),
        "timestamp"               : datetime.utcnow().isoformat() + "Z",
        "total_applications"      : len(batch.applications),
        "portfolio_summary"       : {
            "approved"            : decisions.count("APPROVE"),
            "manual_review"       : decisions.count("MANUAL REVIEW"),
            "rejected"            : decisions.count("REJECT"),
            "approval_rate_pct"   : round(decisions.count("APPROVE") / len(decisions) * 100, 2),
            "rejection_rate_pct"  : round(decisions.count("REJECT")  / len(decisions) * 100, 2),
        },
        "risk_statistics"         : {
            "avg_default_probability" : round(float(prob_array.mean()), 4),
            "min_default_probability" : round(float(prob_array.min()),  4),
            "max_default_probability" : round(float(prob_array.max()),  4),
            "std_default_probability" : round(float(prob_array.std()),  4),
        },
        "health_card_statistics"  : {
            "avg_health_score"    : round(float(health_array.mean()), 1),
            "min_health_score"    : int(health_array.min()),
            "max_health_score"    : int(health_array.max()),
            "pct_above_700"       : round((health_array >= 700).mean() * 100, 2),
            "pct_below_500"       : round((health_array <  500).mean() * 100, 2),
        },
        "risk_band_distribution"  : {
            "GREEN" : sum(1 for p in probabilities if p < 0.35),
            "AMBER" : sum(1 for p in probabilities if 0.35 <= p < 0.65),
            "RED"   : sum(1 for p in probabilities if p >= 0.65),
        }
    }


# ============================================================
# ROUTE 8: Agentic RAG Assessment
# ============================================================

@app.post(
    "/rag/policies/reload",
    tags        = ["RAG"],
    summary     = "Reload policy knowledge base",
    description = "Reload and re-index policy documents from knowledge_base/policies."
)
async def rag_reload_policies():
    count = bootstrap_policy_index(POLICIES_DIR)
    return {
        "status": "ok",
        "policy_chunks": count,
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


@app.post(
    "/rag/analyze",
    tags        = ["RAG"],
    summary     = "Analyze applicant PDF using Agentic RAG",
    description = "Uploads an applicant PDF, runs retrieval + analysis + judge workflow, and returns decision with evidence."
)
async def rag_analyze_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    with NamedTemporaryFile(delete=False, suffix=".pdf", dir=str(UPLOADS_DIR)) as temp_file:
        content = await file.read()
        temp_file.write(content)
        temp_path = Path(temp_file.name)

    try:
        result = run_assessment_workflow(temp_path)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"RAG analysis failed: {str(e)}")
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)


@app.get(
    "/rag/health",
    tags        = ["RAG"],
    summary     = "RAG workflow health",
    description = "Returns simple status for RAG workflow availability."
)
async def rag_health():
    return {
        "status": "healthy",
        "workflow": "LangGraph",
        "retrieval_backend": get_retrieval_backend(),
        "policy_dir": str(POLICIES_DIR),
        "upload_dir": str(UPLOADS_DIR),
        "timestamp": datetime.utcnow().isoformat() + "Z"
    }


# ============================================================
# GLOBAL EXCEPTION HANDLER
# ============================================================

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code = status.HTTP_500_INTERNAL_SERVER_ERROR,
        content     = {
            "error"      : "Internal Server Error",
            "detail"     : str(exc),
            "path"       : str(request.url),
            "timestamp"  : datetime.utcnow().isoformat() + "Z"
        }
    )


# ============================================================
# ROOT ROUTE
# ============================================================

@app.get(
    "/",
    tags    = ["System"],
    summary = "API Root",
    description = "Welcome message and available endpoints."
)
async def root():
    return {
        "message"     : "🏦 MSME Loan Risk Assessment API",
        "version"     : "1.0.0",
        "status"      : "running",
        "docs"        : "/docs",
        "redoc"       : "/redoc",
        "endpoints"   : {
            "GET  /health"          : "API & model health check",
            "GET  /model/info"      : "Model metadata & performance",
            "POST /predict"         : "Single loan application prediction",
            "POST /predict/batch"   : "Batch loan predictions (max 100)",
            "POST /predict/summary" : "Portfolio risk summary statistics",
            "POST /health-card"     : "Financial Health Card only",
            "POST /explain"         : "SHAP explainability only",
            "POST /rag/policies/reload": "Reload policy knowledge base",
            "POST /rag/analyze"     : "Agentic RAG analysis for applicant PDF",
            "GET  /rag/health"      : "RAG workflow health check",
            "GET  /dashboard"       : "Frontend dashboard",
        },
        "timestamp"   : datetime.utcnow().isoformat() + "Z"
    }


@app.get("/dashboard", tags=["UI"], summary="Dashboard")
async def dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not found.")
    return FileResponse(index_path)


# ============================================================
# RUN SERVER
# ============================================================

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(
#         "app:app",
#         host        = "0.0.0.0",
#         port        = 8000,
#         reload      = True,       # Set False in production
#         workers     = 1,          # Increase in production
#         log_level   = "info"
#     )
