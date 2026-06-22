# graph.py
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes.router import router


def generate_answer(state: AgentState) -> dict:
    print(f"[generate_answer] stub — tool used: {state['next_tool']}")
    return {"final_answer": f"Placeholder answer. Tool selected: {state['next_tool']}"}


def route_to_tool(state: AgentState) -> str:
    if state["ambiguous_query"]:
        return END
    return state["next_tool"] or END


builder = StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("generate_answer", generate_answer)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_to_tool, {
    "query_dataset": "generate_answer",
    "retrieve_context": "generate_answer",
    "predict_calories": "generate_answer",
    END: END,
})
builder.add_edge("generate_answer", END)

graph = builder.compile()