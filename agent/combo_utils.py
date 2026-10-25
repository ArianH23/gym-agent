def call_with_retry(fn, state):
    """Call fn(state), retrying once on exception. Returns (result_dict, error_message)."""
    try:
        return fn(state), None
    except Exception:
        try:
            return fn(state), None
        except Exception as e2:
            return None, str(e2)


def merge_tool_results(results):
    """results: dict[tool_name, (result_dict_or_None, error_or_None)].

    Merges successful tools' result dicts (disjoint keys, so no collisions).
    A tool that still failed after retry is recorded in "failed_tools" instead
    of being merged, so a partial failure doesn't block the other tool(s).
    """
    merged = {}
    failed_tools = {}
    for tool_name, (result, error) in results.items():
        if error is not None:
            failed_tools[tool_name] = error
            continue
        merged.update(result)
    merged["failed_tools"] = failed_tools
    return merged
