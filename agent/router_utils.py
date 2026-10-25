VALID_TOOLS = {"query_dataset", "retrieve_context", "predict_calories"}


def parse_router_decision(raw_response):
    decision = raw_response.strip().lower()
    tokens = [token.strip() for token in decision.split("+") if token.strip()]
    valid_tokens = sorted({token for token in tokens if token in VALID_TOOLS})

    if not valid_tokens:
        return {"next_tool": None, "ambiguous_query": True}

    return {"next_tool": "+".join(valid_tokens), "ambiguous_query": False}
