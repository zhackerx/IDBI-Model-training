from collections import Counter


SECTION_RULES = {
    "income_tax": {
        "keywords": ["itr", "income tax", "return", "filing"],
        "negative": ["missing", "not filed", "default", "penalty"],
    },
    "gst": {
        "keywords": ["gst", "gstr", "turnover", "tax invoice"],
        "negative": ["delay", "non-compliance", "default"],
    },
    "credit_history": {
        "keywords": ["cibil", "credit score", "dpd", "default", "delinquency"],
        "negative": ["written off", "settled", "overdue", "default"],
    },
    "existing_loans": {
        "keywords": ["loan", "emi", "liability", "debt"],
        "negative": ["high emi", "overdue", "stressed"],
    },
    "bank_details": {
        "keywords": ["bank", "cash flow", "balance", "bounce"],
        "negative": ["bounce", "low balance", "irregular"],
    },
}


def _section_assessment(section_name: str, evidence_chunks: list[dict]) -> dict:
    rules = SECTION_RULES[section_name]
    text_blob = " ".join([c["text"] for c in evidence_chunks]).lower()

    keyword_hits = sum(1 for k in rules["keywords"] if k in text_blob)
    negative_hits = sum(1 for k in rules["negative"] if k in text_blob)

    if keyword_hits == 0:
        status = "NEEDS_REVIEW"
        confidence = 0.45
        reason = "Insufficient evidence retrieved for this section."
    elif negative_hits >= 2:
        status = "HIGH_RISK"
        confidence = 0.8
        reason = "Multiple risk markers detected in retrieved evidence."
    elif negative_hits == 1:
        status = "NEEDS_REVIEW"
        confidence = 0.65
        reason = "Some risk markers detected; requires deeper verification."
    else:
        status = "PASS"
        confidence = 0.8
        reason = "Retrieved evidence does not show major policy violations."

    top_evidence = [c["text"][:180] for c in evidence_chunks[:2]]

    return {
        "status": status,
        "confidence": confidence,
        "reason": reason,
        "evidence": top_evidence,
    }


def analyze_sections(retrieved: dict[str, list[dict]]) -> dict:
    """Builds section-wise underwriting analysis from retrieved chunks."""
    section_results = {
        section: _section_assessment(section, chunks)
        for section, chunks in retrieved.items()
    }

    status_counter = Counter(result["status"] for result in section_results.values())

    return {
        "sections": section_results,
        "summary": {
            "pass_count": status_counter.get("PASS", 0),
            "needs_review_count": status_counter.get("NEEDS_REVIEW", 0),
            "high_risk_count": status_counter.get("HIGH_RISK", 0),
        },
    }
