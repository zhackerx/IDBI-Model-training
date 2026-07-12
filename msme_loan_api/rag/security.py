import re
from dataclasses import dataclass, field


MAX_UPLOAD_BYTES = 10 * 1024 * 1024
ALLOWED_RAG_ROLE = "credit_officer"
ACCESS_SCOPE = "msme_loan_underwriting"


@dataclass
class UploadSecurityScan:
    accepted: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


PII_PATTERNS = [
    (re.compile(r"\b[A-Z]{5}[0-9]{4}[A-Z]\b", re.IGNORECASE), "[REDACTED_PAN]"),
    (re.compile(r"\b[0-9]{2}[A-Z]{5}[0-9]{4}[A-Z][1-9A-Z]Z[0-9A-Z]\b", re.IGNORECASE), "[REDACTED_GSTIN]"),
    (re.compile(r"\b(?:\d[ -]?){12}\b"), "[REDACTED_ID_NUMBER]"),
    (re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD_OR_ACCOUNT]"),
    (re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE), "[REDACTED_EMAIL]"),
    (re.compile(r"(?<!\d)(?:\+91[ -]?)?[6-9]\d{9}(?!\d)"), "[REDACTED_PHONE]"),
]

HAZARDOUS_DIRECTIVE_PATTERNS = [
    re.compile(r"ignore\s+(?:all\s+)?previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(?:all\s+)?(?:prior|previous)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(?:the\s+)?(?:system|developer)\s+prompt", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+(?:in\s+)?developer\s+mode", re.IGNORECASE),
    re.compile(r"exfiltrate|bypass\s+policy|disable\s+guardrails", re.IGNORECASE),
]


def scan_pdf_upload(filename: str, content: bytes) -> UploadSecurityScan:
    errors = []
    warnings = []
    normalized_name = (filename or "").lower()

    if not normalized_name.endswith(".pdf"):
        errors.append("Only PDF files are accepted.")
    if len(content) == 0:
        errors.append("Uploaded PDF is empty.")
    if len(content) > MAX_UPLOAD_BYTES:
        errors.append(f"Uploaded PDF exceeds the {MAX_UPLOAD_BYTES // (1024 * 1024)} MB limit.")
    if content and not content.startswith(b"%PDF"):
        errors.append("Uploaded file does not have a valid PDF signature.")

    if b"/JavaScript" in content or b"/JS" in content:
        warnings.append("PDF contains JavaScript markers; active content will not be executed.")

    return UploadSecurityScan(accepted=not errors, errors=errors, warnings=warnings)


def sanitize_text_for_rag(text: str) -> tuple[str, dict]:
    sanitized = text or ""
    pii_redactions = 0
    stripped_directives = 0

    for pattern, replacement in PII_PATTERNS:
        sanitized, count = pattern.subn(replacement, sanitized)
        pii_redactions += count

    for pattern in HAZARDOUS_DIRECTIVE_PATTERNS:
        sanitized, count = pattern.subn("[REMOVED_PROMPT_DIRECTIVE]", sanitized)
        stripped_directives += count

    return sanitized, {
        "pii_redactions": pii_redactions,
        "stripped_prompt_directives": stripped_directives,
    }


def secured_chunk_metadata(source: str, doc_type: str, chunk_id: int) -> dict:
    return {
        "source": source,
        "type": doc_type,
        "chunk_id": chunk_id,
        "access_role": ALLOWED_RAG_ROLE,
        "access_scope": ACCESS_SCOPE,
        "pii_state": "redacted_before_embedding",
    }
