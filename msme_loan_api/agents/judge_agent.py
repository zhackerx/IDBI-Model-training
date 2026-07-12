def judge_reasoning(analysis: dict) -> dict:
    """Judge node validates evidence coverage and consistency."""
    sections = analysis.get("sections", {})
    missing_evidence = []
    low_confidence = []

    for section, result in sections.items():
        evidence = result.get("evidence", [])
        confidence = result.get("confidence", 0.0)

        if not evidence:
            missing_evidence.append(section)
        if confidence < 0.6:
            low_confidence.append(section)

    contradictions = []
    for section, result in sections.items():
        if result.get("status") == "PASS" and "risk" in result.get("reason", "").lower():
            contradictions.append(section)

    approved = len(missing_evidence) == 0 and len(low_confidence) <= 1 and len(contradictions) == 0

    return {
        "approved": approved,
        "missing_evidence": missing_evidence,
        "low_confidence_sections": low_confidence,
        "contradictions": contradictions,
        "reason": "Reasoning approved" if approved else "Reasoning needs corrective retrieval",
    }
