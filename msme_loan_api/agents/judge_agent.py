def _normalize_evidence(text: str) -> str:
    return " ".join(text.lower().split())[:220]


def judge_reasoning(analysis: dict) -> dict:
    """Judge node validates evidence coverage and consistency."""
    sections = analysis.get("sections", {})
    missing_evidence = []
    low_confidence = []
    repeated_evidence_sections = set()

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

    evidence_to_sections: dict[str, set[str]] = {}
    for section, result in sections.items():
        for ev in result.get("evidence", []):
            key = _normalize_evidence(ev)
            if not key:
                continue
            evidence_to_sections.setdefault(key, set()).add(section)

    for sec_set in evidence_to_sections.values():
        # If the same evidence snippet supports many sections,
        # section grounding is weak and should not pass the judge gate.
        if len(sec_set) >= 3:
            repeated_evidence_sections.update(sec_set)

    approved = (
        len(missing_evidence) == 0
        and len(low_confidence) <= 1
        and len(contradictions) == 0
    )

    if not approved:
        gate_status = "RETRY_REQUIRED"
        quality_score = 0.72
    elif repeated_evidence_sections:
        gate_status = "APPROVED_WITH_WARNINGS"
        quality_score = 0.82
    else:
        gate_status = "APPROVED"
        quality_score = 0.90

    return {
        "approved": approved,
        "gate_status": gate_status,
        "quality_score": quality_score,
        "missing_evidence": missing_evidence,
        "low_confidence_sections": low_confidence,
        "contradictions": contradictions,
        "repeated_evidence_sections": sorted(repeated_evidence_sections),
        "reason": "Reasoning approved" if approved else "Reasoning needs corrective retrieval",
    }
