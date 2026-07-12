from rag.retriever import RetrievedChunk, Retriever


QUERY_EXPANSIONS = {
    "itr": "income tax return filing trend",
    "gst": "gst return filing turnover compliance",
    "credit": "cibil credit score defaults dpd",
    "loan": "existing loan emi debt obligations",
    "bank": "bank statement cashflow bounce average balance",
}


def corrective_retrieve(
    retriever: Retriever,
    query: str,
    top_k: int = 6,
    min_score: float = 0.05,
) -> tuple[list[RetrievedChunk], dict]:
    """Retry retrieval with query expansion when confidence is low."""
    first_pass = retriever.search(query, top_k=top_k)
    top_score = first_pass[0].score if first_pass else 0.0

    diagnostics = {
        "first_pass_top_score": top_score,
        "expanded": False,
    }

    if top_score >= min_score:
        return first_pass, diagnostics

    expanded_query = query
    for key, expansion in QUERY_EXPANSIONS.items():
        if key in query.lower():
            expanded_query = f"{query} {expansion}"

    second_pass = retriever.search(expanded_query, top_k=top_k)
    diagnostics["expanded"] = True
    diagnostics["expanded_query"] = expanded_query
    diagnostics["second_pass_top_score"] = second_pass[0].score if second_pass else 0.0

    return second_pass if second_pass else first_pass, diagnostics
