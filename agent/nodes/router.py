from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.prompts import ROUTER_PROMPT
from agent.router_utils import parse_router_decision
from agent.state import AgentState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0)


def router(state: AgentState) -> dict:
    question = state["messages"][-1].content
    print(f"[router] classifying: {question}")

    response = llm.invoke([
        SystemMessage(content=ROUTER_PROMPT),
        HumanMessage(content=question)
    ])

    result = parse_router_decision(response.content)

    if result["ambiguous_query"]:
        print(f"[router] unexpected response '{response.content.strip().lower()}', marking ambiguous")
    else:
        print(f"[router] decision: {result['next_tool']}")

    return result
