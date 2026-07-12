from collections import Counter
import re


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


def _count_negative_hits(text: str, markers: list[str]) -> int:
    """Count risk markers while ignoring simple negations like 'no default'."""
    hits = 0
    text_l = text.lower()

    for marker in markers:
        marker_l = marker.lower()
        if marker_l not in text_l:
            continue

        # Skip when a marker is explicitly negated in nearby phrase.
        negation_patterns = [
            rf"\bno\s+{re.escape(marker_l)}\b",
            rf"\bnot\s+{re.escape(marker_l)}\b",
            rf"\bwithout\s+{re.escape(marker_l)}\b",
            rf"\bzero\s+{re.escape(marker_l)}\b",
        ]
        if any(re.search(pattern, text_l) for pattern in negation_patterns):
            continue

        hits += 1

    return hits


def _section_assessment(section_name: str, evidence_chunks: list[dict]) -> dict:
    rules = SECTION_RULES[section_name]

    applicant_text = " ".join(
        [c.get("text", "") for c in evidence_chunks if c.get("metadata", {}).get("type") == "applicant"]
    ).lower()
    policy_text = " ".join(
        [c.get("text", "") for c in evidence_chunks if c.get("metadata", {}).get("type") == "policy"]
    ).lower()
    text_blob = (applicant_text + " " + policy_text).strip()

    keyword_hits = sum(1 for k in rules["keywords"] if k in text_blob)

    # Critical: evaluate risk markers from applicant evidence only.
    # Policy chunks naturally contain words like "default" and should not
    # by themselves increase applicant risk.
    negative_hits = _count_negative_hits(applicant_text, rules["negative"])

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
