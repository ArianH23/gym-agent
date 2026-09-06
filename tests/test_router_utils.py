from agent.router_utils import parse_router_decision


def test_query_dataset_decision_is_recognized():
    result = parse_router_decision("query_dataset")
    assert result == {"next_tool": "query_dataset", "ambiguous_query": False}


def test_retrieve_context_decision_is_recognized():
    result = parse_router_decision("retrieve_context")
    assert result == {"next_tool": "retrieve_context", "ambiguous_query": False}


def test_predict_calories_decision_is_recognized():
    result = parse_router_decision("predict_calories")
    assert result == {"next_tool": "predict_calories", "ambiguous_query": False}


def test_decision_with_whitespace_and_mixed_case_is_normalized():
    result = parse_router_decision("  Query_Dataset\n")
    assert result == {"next_tool": "query_dataset", "ambiguous_query": False}


def test_unrecognized_decision_is_marked_ambiguous():
    result = parse_router_decision("i'm not sure, maybe query the data?")
    assert result == {"next_tool": None, "ambiguous_query": True}


def test_empty_response_is_marked_ambiguous():
    result = parse_router_decision("")
    assert result == {"next_tool": None, "ambiguous_query": True}
