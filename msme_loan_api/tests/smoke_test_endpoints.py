import argparse
import json
from pathlib import Path

import requests


DEFAULT_PREDICT_PAYLOAD = {
    "annual_income": 45000,
    "emp_length": 3,
    "application_type": "individual",
    "loan_amount": 15000,
    "term": 36,
    "int_rate": 14.5,
    "installment": 350.0,
    "purpose": "car",
    "grade": "c",
    "sub_grade": "c4",
    "home_ownership": "rent",
    "verification_status": "source verified",
    "dti": 18.5,
    "total_acc": 8,
    "total_payment": 5000,
    "address_state": "CA",
    "loan_age_months": 12,
    "issue_month": 3,
    "issue_year": 2021,
    "days_since_last_payment": 90,
    "days_since_credit_pull": 30,
    "days_to_next_payment": 15,
}


def _check(resp: requests.Response, name: str) -> None:
    if not resp.ok:
        raise RuntimeError(f"{name} failed with status {resp.status_code}: {resp.text[:300]}")


def run_smoke_tests(base_url: str, pdf_path: Path) -> dict:
    summary = {}

    health = requests.get(f"{base_url}/health", timeout=20)
    _check(health, "/health")
    summary["/health"] = health.json()

    rag_health = requests.get(f"{base_url}/rag/health", timeout=20)
    _check(rag_health, "/rag/health")
    summary["/rag/health"] = rag_health.json()

    reload_resp = requests.post(f"{base_url}/rag/policies/reload", timeout=30)
    _check(reload_resp, "/rag/policies/reload")
    summary["/rag/policies/reload"] = reload_resp.json()

    with pdf_path.open("rb") as f:
        rag_resp = requests.post(
            f"{base_url}/rag/analyze",
            files={"file": (pdf_path.name, f, "application/pdf")},
            timeout=60,
        )
    _check(rag_resp, "/rag/analyze")
    summary["/rag/analyze"] = rag_resp.json()

    predict_resp = requests.post(
        f"{base_url}/predict",
        json=DEFAULT_PREDICT_PAYLOAD,
        timeout=60,
    )
    _check(predict_resp, "/predict")
    summary["/predict"] = predict_resp.json()

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Quick smoke tests for hackathon API endpoints")
    parser.add_argument("--base-url", default="http://localhost:8000", help="Base URL of running API")
    parser.add_argument(
        "--pdf",
        default=str(Path(__file__).resolve().parents[1] / "uploads" / "sample_applicant_low_risk.pdf"),
        help="Path to applicant PDF for /rag/analyze",
    )
    parser.add_argument("--out", default="", help="Optional path to write JSON report")

    args = parser.parse_args()
    pdf_path = Path(args.pdf)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    results = run_smoke_tests(args.base_url.rstrip("/"), pdf_path)

    print("Smoke tests passed for endpoints:")
    for endpoint in ["/health", "/rag/health", "/rag/policies/reload", "/rag/analyze", "/predict"]:
        print(f"- {endpoint}")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"\nSaved report to: {out_path}")


if __name__ == "__main__":
    main()
