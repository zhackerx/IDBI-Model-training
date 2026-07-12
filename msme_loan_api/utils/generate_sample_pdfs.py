from pathlib import Path


def _pdf_escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def write_simple_pdf(lines: list[str], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stream_lines = [
        "BT",
        "/F1 11 Tf",
        "72 790 Td",
    ]

    first = True
    for line in lines:
        safe = _pdf_escape(line)
        if first:
            stream_lines.append(f"({safe}) Tj")
            first = False
        else:
            stream_lines.append("0 -14 Td")
            stream_lines.append(f"({safe}) Tj")

    stream_lines.append("ET")
    stream_data = "\n".join(stream_lines).encode("utf-8")

    objects: list[bytes] = []
    objects.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    objects.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    objects.append(
        b"3 0 obj\n"
        b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 595 842] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\n"
        b"endobj\n"
    )
    objects.append(
        f"4 0 obj\n<< /Length {len(stream_data)} >>\nstream\n".encode("utf-8")
        + stream_data
        + b"\nendstream\nendobj\n"
    )
    objects.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")

    header = b"%PDF-1.4\n"
    offsets = [0]
    content = bytearray(header)

    for obj in objects:
        offsets.append(len(content))
        content.extend(obj)

    xref_start = len(content)
    content.extend(f"xref\n0 {len(offsets)}\n".encode("utf-8"))
    content.extend(b"0000000000 65535 f \n")

    for off in offsets[1:]:
        content.extend(f"{off:010d} 00000 n \n".encode("utf-8"))

    content.extend(
        (
            "trailer\n"
            f"<< /Size {len(offsets)} /Root 1 0 R >>\n"
            "startxref\n"
            f"{xref_start}\n"
            "%%EOF\n"
        ).encode("utf-8")
    )

    output_path.write_bytes(bytes(content))


def main() -> None:
    uploads = Path(__file__).resolve().parents[1] / "uploads"

    low_risk_lines = [
        "IDBI MSME Loan Applicant Dossier - LOW RISK",
        "Applicant Name: Shree Precision Components LLP",
        "PAN: AAPFS3478K | GSTIN: 27AAPFS3478K1ZS",
        "Business Vintage: 8 years",
        "",
        "ITR SUMMARY",
        "- ITR available for FY 2021-22, 2022-23, 2023-24",
        "- Income trend stable and growing",
        "- No tax defaults or penalties",
        "",
        "GST SUMMARY",
        "- GST active and compliant",
        "- On-time filings for last 8 quarters",
        "- Quarterly turnover INR 62,00,000 average",
        "",
        "CREDIT HISTORY",
        "- CIBIL score: 781",
        "- No written-off or settled accounts",
        "- DPD in last 24 months: 0",
        "",
        "EXISTING LOANS",
        "- Existing EMI INR 70,000/month",
        "- Debt-to-income ratio within policy threshold",
        "",
        "BANK STATEMENT INSIGHTS",
        "- Stable monthly cashflow",
        "- Cheque bounce count last 12 months: 0",
        "- Strong average balance",
        "",
        "RECOMMENDATION CANDIDATE",
        "- APPROVE",
    ]

    high_risk_lines = [
        "IDBI MSME Loan Applicant Dossier - HIGH RISK",
        "Applicant Name: Rapid Build Traders",
        "PAN: BDFPR9123Q | GSTIN: 27BDFPR9123Q1Z4",
        "Business Vintage: 2 years",
        "",
        "ITR SUMMARY",
        "- ITR missing for FY 2022-23",
        "- Income inconsistent across years",
        "- Tax penalty noted in records",
        "",
        "GST SUMMARY",
        "- Multiple delayed GST filings in last 12 months",
        "- GST turnover volatile and declining",
        "",
        "CREDIT HISTORY",
        "- CIBIL score: 592",
        "- Two accounts settled",
        "- 60+ DPD events in last 18 months",
        "",
        "EXISTING LOANS",
        "- Existing EMI INR 1,85,000/month",
        "- Debt obligations exceed internal threshold",
        "",
        "BANK STATEMENT INSIGHTS",
        "- Frequent balance dips",
        "- Cheque bounce count last 6 months: 7",
        "- Irregular inflow pattern",
        "",
        "RISK FLAGS",
        "- High delinquency and compliance concerns",
        "",
        "RECOMMENDATION CANDIDATE",
        "- REJECT",
    ]

    write_simple_pdf(low_risk_lines, uploads / "sample_applicant_low_risk.pdf")
    write_simple_pdf(high_risk_lines, uploads / "sample_applicant_high_risk.pdf")


if __name__ == "__main__":
    main()
