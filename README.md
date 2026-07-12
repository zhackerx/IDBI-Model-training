# IDBI MSME Credit Assessment - Agentic RAG + ML API

This repository contains a hackathon-ready MSME credit assessment solution with:

1. ML risk scoring API (FastAPI + model artifacts)
2. Agentic RAG workflow (PDF parsing, retrieval, analysis agent, judge agent, corrective retrieval)
3. Switchable retrieval backend (TF-IDF or Chroma vector DB)
4. Simple dashboard for PDF upload and analysis
5. Smoke test script for reliability checks

## Problem Context

The solution targets MSME loan evaluation for credit-invisible or thin-file applicants by combining:

1. Financial Health Card scoring
2. Explainable loan risk prediction
3. Evidence-based policy validation from applicant PDF + policy knowledge base

The objective is recommendation support, not autonomous sanctioning.

## Repository Structure

1. `msme_loan_api/`
	FastAPI app, RAG workflow, dashboard, policy KB, tests, model-serving endpoints.
2. `msme_model_training/`
	Data cleansing and model training pipelines with generated artifacts/plots.

## Core Features Implemented

1. ML endpoints: `/predict`, `/predict/batch`, `/predict/summary`, `/health-card`, `/explain`, `/health`
2. RAG endpoints: `/rag/health`, `/rag/policies/reload`, `/rag/analyze`
3. Dashboard: `/dashboard`
4. LangGraph workflow with analysis and judge nodes
5. Corrective retrieval loop for low-confidence evidence
6. Sample applicant PDFs (normal, low risk, high risk)

## Quick Start (Local)

### 1. Create and activate virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
cd msme_loan_api
pip install -r requirements.txt
```

### 3. Ensure model files exist

Place these in `msme_loan_api/model/`:

1. `best_model.pkl`
2. `scaler.pkl`
3. `model_metadata.pkl`

### 4. Run API (default TF-IDF retrieval)

```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open:

1. Swagger: `http://localhost:8000/docs`
2. Dashboard: `http://localhost:8000/dashboard`

## Enable Chroma Vector DB Mode

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

Verify backend:

1. Call `GET /rag/health`
2. Check `retrieval_backend` in response

## Demo PDFs

Use these files in `msme_loan_api/uploads/` for fast demos:

1. `sample_applicant_test.pdf`
2. `sample_applicant_low_risk.pdf`
3. `sample_applicant_high_risk.pdf`

## Endpoint Smoke Test

From `msme_loan_api/` while API is running:

```bash
python tests/smoke_test_endpoints.py --base-url http://localhost:8000
```

This validates:

1. `/health`
2. `/rag/health`
3. `/rag/policies/reload`
4. `/rag/analyze`
5. `/predict`

Optional report output:

```bash
python tests/smoke_test_endpoints.py --base-url http://localhost:8000 --out smoke_report.json
```

## Notes

1. If `origin/main` and local `main` diverge with unrelated history, merge with care before push.
2. Large CSV artifacts exist in training outputs; consider Git LFS for long-term maintenance.
3. For deeper API field-level payload docs, see `msme_loan_api/README.md`.