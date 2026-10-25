from agent.combo_utils import call_with_retry, merge_tool_results


def test_call_with_retry_returns_result_on_first_success():
    result, error = call_with_retry(lambda state: {"ok": True}, {})
    assert result == {"ok": True}
    assert error is None


def test_call_with_retry_retries_once_then_succeeds():
    attempts = []

    def flaky(state):
        attempts.append(1)
        if len(attempts) == 1:
            raise ValueError("transient")
        return {"ok": True}

    result, error = call_with_retry(flaky, {})
    assert result == {"ok": True}
    assert error is None
    assert len(attempts) == 2


def test_call_with_retry_returns_error_after_second_failure():
    def always_fails(state):
        raise ValueError("permanent")

    result, error = call_with_retry(always_fails, {})
    assert result is None
    assert error == "permanent"


def test_merge_tool_results_combines_disjoint_dicts():
    results = {
        "predict_calories": ({"predicted_calories": 500.0}, None),
        "retrieve_context": ({"retrieved_context": ["doc"]}, None),
    }
    merged = merge_tool_results(results)
    assert merged["predicted_calories"] == 500.0
    assert merged["retrieved_context"] == ["doc"]
    assert merged["failed_tools"] == {}


def test_merge_tool_results_records_failed_tool_without_merging_its_data():
    results = {
        "predict_calories": (None, "network timeout"),
        "retrieve_context": ({"retrieved_context": ["doc"]}, None),
    }
    merged = merge_tool_results(results)
    assert "predicted_calories" not in merged
    assert merged["retrieved_context"] == ["doc"]
    assert merged["failed_tools"] == {"predict_calories": "network timeout"}


def test_merge_tool_results_with_all_failures_returns_only_failed_tools():
    results = {
        "predict_calories": (None, "boom"),
        "retrieve_context": (None, "also boom"),
    }
    merged = merge_tool_results(results)
    assert merged == {
        "failed_tools": {"predict_calories": "boom", "retrieve_context": "also boom"}
    }
