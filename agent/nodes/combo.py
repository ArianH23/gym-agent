from concurrent.futures import ThreadPoolExecutor

from agent.combo_utils import call_with_retry, merge_tool_results
from agent.nodes.predict_calories import predict_calories
from agent.nodes.query_dataset import query_dataset
from agent.nodes.retrieve_context import retrieve_context
from agent.state import AgentState

TOOL_FUNCTIONS = {
    "query_dataset": query_dataset,
    "predict_calories": predict_calories,
    "retrieve_context": retrieve_context,
}


def run_combo(state: AgentState) -> dict:
    tool_names = state["next_tool"].split("+")
    print(f"[combo] running tools in parallel: {tool_names}")

    with ThreadPoolExecutor(max_workers=len(tool_names)) as executor:
        futures = {
            name: executor.submit(call_with_retry, TOOL_FUNCTIONS[name], state)
            for name in tool_names
        }
        results = {name: future.result() for name, future in futures.items()}

    merged = merge_tool_results(results)

    if merged["failed_tools"]:
        print(f"[combo] permanently failed after retry: {merged['failed_tools']}")

    return merged
