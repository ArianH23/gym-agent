from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.prompts import ROUTER_PROMPT
from agent.state import AgentState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


def router(state: AgentState) -> dict:
    question = state["messages"][-1].content
    print(f"[router] classifying: {question}")

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=question)
    ])

    decision = response.content.strip().lower()

    valid_tools = {"query_dataset", "retrieve_context", "predict_calories"}
    if decision not in valid_tools:
        print(f"[router] unexpected response '{decision}', marking ambiguous")
        return {"next_tool": None, "ambiguous_query": True}

    print(f"[router] decision: {decision}")
    return {"next_tool": decision, "ambiguous_query": False}
