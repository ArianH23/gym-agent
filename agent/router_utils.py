VALID_TOOLS = {"query_dataset", "retrieve_context", "predict_calories"}


def parse_router_decision(raw_response):
    decision = raw_response.strip().lower()
    if decision not in VALID_TOOLS:
        return {"next_tool": None, "ambiguous_query": True}
    return {"next_tool": decision, "ambiguous_query": False}
