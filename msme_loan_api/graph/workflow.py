from pathlib import Path
import os
from typing import TypedDict

from langgraph.graph import END, StateGraph

from agents.analysis_agent import analyze_sections
from agents.judge_agent import judge_reasoning
from rag.corrective_rag import corrective_retrieve
from rag.parser import chunk_text, extract_text_from_pdf
from rag.retriever import RetrievedChunk, Retriever, TfidfRetriever
from vector_store.chroma_store import ChromaRetriever


class WorkflowState(TypedDict, total=False):
    applicant_text: str
    applicant_chunks: list[str]
    applicant_retriever: Retriever
    retrieved: dict[str, list[dict]]
    analysis: dict
    judge: dict
    retries: int
    corrective_diagnostics: dict
    final: dict


SECTION_QUERIES = {
    "income_tax": "itr income tax return filing consistency",
    "gst": "gst filing turnover compliance",
    "credit_history": "cibil credit score defaults dpd delinquencies",
    "existing_loans": "existing loans emi liabilities debt ratio",
    "bank_details": "bank statement cash flow bounce average balance",
}

SECTION_TERMS = {
    "income_tax": ["itr", "income tax", "return", "tax"],
    "gst": ["gst", "gstr", "turnover"],
    "credit_history": ["cibil", "credit", "dpd", "delinqu"],
    "existing_loans": ["loan", "emi", "liabilit", "debt"],
    "bank_details": ["bank", "cash flow", "balance", "bounce"],
}


RAG_BACKEND = os.getenv("RAG_BACKEND", "tfidf").strip().lower()
CHROMA_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", Path(__file__).resolve().parents[1] / "chroma_db"))


def _new_retriever(collection_name: str) -> Retriever:
    if RAG_BACKEND == "chroma":
        try:
            return ChromaRetriever(collection_name=collection_name, persist_dir=CHROMA_DIR)
        except Exception:
            # Keep service usable when Chroma is unavailable.
            return TfidfRetriever()
    return TfidfRetriever()


_policy_retriever = _new_retriever("policy_kb")


def _to_dict_chunks(chunks: list[RetrievedChunk]) -> list[dict]:
    return [
        {
            "text": chunk.text,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
        for chunk in chunks
    ]


def _norm_text(text: str) -> str:
    return " ".join(text.lower().split())[:240]


def _section_retrieval(section: str, app_hits: list[RetrievedChunk], pol_hits: list[RetrievedChunk]) -> list[dict]:
    terms = SECTION_TERMS.get(section, [])

    # Combine and sort by relevance, then deduplicate by normalized text.
    combined = sorted(app_hits + pol_hits, key=lambda c: c.score, reverse=True)
    unique: list[RetrievedChunk] = []
    seen = set()
    for chunk in combined:
        key = _norm_text(chunk.text)
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)

    def is_relevant(chunk: RetrievedChunk) -> bool:
        text = chunk.text.lower()
        return any(term in text for term in terms)

    relevant = [c for c in unique if is_relevant(c)]
    selected = relevant[:4] if relevant else unique[:4]
    return _to_dict_chunks(selected)


def bootstrap_policy_index(policy_dir: Path) -> int:
    docs: list[str] = []
    metas: list[dict] = []

    supported = {".txt", ".md", ".pdf"}
    for path in policy_dir.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in supported:
            continue

        if path.suffix.lower() == ".pdf":
            raw_text = extract_text_from_pdf(path)
        else:
            raw_text = path.read_text(encoding="utf-8", errors="ignore")

        chunks = chunk_text(raw_text)
        for idx, ch in enumerate(chunks):
            docs.append(ch)
            metas.append({"source": str(path.name), "type": "policy", "chunk_id": idx})

    return _policy_retriever.index_documents(docs, metas)


def _retrieve_node(state: WorkflowState) -> WorkflowState:
    applicant_docs = state.get("applicant_chunks", [])
    applicant_meta = [
        {"source": "applicant_pdf", "type": "applicant", "chunk_id": idx}
        for idx in range(len(applicant_docs))
    ]
    applicant_retriever = _new_retriever("applicant_docs")
    applicant_retriever.index_documents(applicant_docs, applicant_meta)

    retrieved = {}
    for section, query in SECTION_QUERIES.items():
        app_hits = applicant_retriever.search(query, top_k=6)
        pol_hits = _policy_retriever.search(query, top_k=6)
        retrieved[section] = _section_retrieval(section, app_hits, pol_hits)

    return {
        "applicant_retriever": applicant_retriever,
        "retrieved": retrieved,
    }


def _analysis_node(state: WorkflowState) -> WorkflowState:
    return {"analysis": analyze_sections(state.get("retrieved", {}))}


def _judge_node(state: WorkflowState) -> WorkflowState:
    return {"judge": judge_reasoning(state.get("analysis", {}))}


def _corrective_node(state: WorkflowState) -> WorkflowState:
    retries = state.get("retries", 0) + 1
    applicant_retriever = state["applicant_retriever"]

    retrieved = {}
    diagnostics = {}
    for section, query in SECTION_QUERIES.items():
        app_hits, diag = corrective_retrieve(applicant_retriever, query, top_k=8)
        pol_hits, _ = corrective_retrieve(_policy_retriever, query, top_k=6)
        retrieved[section] = _section_retrieval(section, app_hits, pol_hits)
        diagnostics[section] = diag

    return {
        "retrieved": retrieved,
        "retries": retries,
        "corrective_diagnostics": diagnostics,
    }


def _route_after_judge(state: WorkflowState) -> str:
    judge = state.get("judge", {})
    retries = state.get("retries", 0)
    if judge.get("approved") or retries >= 1:
        return "finalize"
    return "corrective"


def _finalize_node(state: WorkflowState) -> WorkflowState:
    analysis = state.get("analysis", {})
    summary = analysis.get("summary", {})
    judge = state.get("judge", {})

    if summary.get("high_risk_count", 0) > 0:
        base_decision = "Reject"
        base_risk_score = 82
    elif summary.get("needs_review_count", 0) > 0:
        base_decision = "Conditional Approval"
        base_risk_score = 63
    else:
        base_decision = "Approve"
        base_risk_score = 28

    # Judge is a reasoning-quality gate. If it fails, force human review.
    if not judge.get("approved", False):
        decision = "Manual Review"
        risk_score = max(base_risk_score, 70 if summary.get("high_risk_count", 0) > 0 else 55)
    else:
        decision = base_decision
        risk_score = base_risk_score

    confidence = float(judge.get("quality_score", 0.72))

    evidence = []
    for section, result in analysis.get("sections", {}).items():
        for ev in result.get("evidence", [])[:1]:
            evidence.append(f"{section}: {ev}")

    final = {
        "decision": decision,
        "base_decision": base_decision,
        "risk_score": risk_score,
        "confidence": confidence,
        "summary": analysis.get("summary", {}),
        "sections": analysis.get("sections", {}),
        "judge": judge,
        "judge_gate_status": judge.get("gate_status", "UNKNOWN"),
        "evidence": evidence[:8],
        "corrective_diagnostics": state.get("corrective_diagnostics", {}),
    }

    return {"final": final}


def _build_workflow():
    graph = StateGraph(WorkflowState)
    graph.add_node("retrieve_step", _retrieve_node)
    graph.add_node("analyze_step", _analysis_node)
    graph.add_node("judge_step", _judge_node)
    graph.add_node("corrective_step", _corrective_node)
    graph.add_node("finalize_step", _finalize_node)

    graph.set_entry_point("retrieve_step")
    graph.add_edge("retrieve_step", "analyze_step")
    graph.add_edge("analyze_step", "judge_step")
    graph.add_conditional_edges(
        "judge_step",
        _route_after_judge,
        {
            "corrective": "corrective_step",
            "finalize": "finalize_step",
        },
    )
    graph.add_edge("corrective_step", "analyze_step")
    graph.add_edge("finalize_step", END)

    return graph.compile()


_workflow = _build_workflow()


def run_assessment_workflow(pdf_path: Path) -> dict:
    applicant_text = extract_text_from_pdf(pdf_path)
    applicant_chunks = chunk_text(applicant_text)

    state: WorkflowState = {
        "applicant_text": applicant_text,
        "applicant_chunks": applicant_chunks,
        "retries": 0,
    }

    result = _workflow.invoke(state)
    return result.get("final", {})


def get_retrieval_backend() -> str:
    if RAG_BACKEND == "chroma" and isinstance(_policy_retriever, ChromaRetriever):
        return "chroma"
    return "tfidf"
