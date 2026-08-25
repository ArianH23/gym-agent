# graph.py
from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, START, END
from agent.state import AgentState
from agent.nodes.router import router
from agent.nodes.query_dataset import query_dataset
from agent.nodes.predict_calories import predict_calories
from agent.nodes.retrieve_context import retrieve_context
from agent.nodes.generate_answer import generate_answer


# def generate_answer(state: AgentState) -> dict:
#     print(f"[generate_answer] stub — tool used: {state['next_tool']}")
#     return {"final_answer": f"Placeholder answer. Tool selected: {state['next_tool']}"}


def route_to_tool(state: AgentState) -> str:
    if state["ambiguous_query"]:
        return END
    return state["next_tool"] or END


builder = StateGraph(AgentState)

builder.add_node("router", router)
builder.add_node("generate_answer", generate_answer)
builder.add_node("query_dataset", query_dataset)
builder.add_node("predict_calories", predict_calories)
builder.add_node("retrieve_context", retrieve_context)

builder.add_edge(START, "router")
builder.add_conditional_edges("router", route_to_tool, {
    "query_dataset": "query_dataset",
    "retrieve_context": "retrieve_context",
    "predict_calories": "predict_calories",
    END: END,
})
builder.add_edge("query_dataset", "generate_answer")
builder.add_edge("predict_calories", "generate_answer")
builder.add_edge("retrieve_context", "generate_answer")
builder.add_edge("generate_answer", END)

graph = builder.compile()