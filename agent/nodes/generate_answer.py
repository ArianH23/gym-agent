from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

from agent.context_builder import build_context
from agent.state import AgentState

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.3)

ANSWER_PROMPT = """You are a helpful fitness assistant.
Synthesize the provided data into a clear, concise natural language response.
Be specific with numbers. Don't make up information not present in the data.
Keep the response conversational and helpful, 2-4 sentences maximum."""


def generate_answer(state: AgentState) -> dict:
    question = state["messages"][-1].content
    next_tool = state["next_tool"]

    context = build_context(
        next_tool,
        dataset_results=state.get("dataset_results"),
        predicted_calories=state.get("predicted_calories"),
        prediction_inputs=state.get("prediction_inputs"),
        retrieved_context=state.get("retrieved_context"),
    )

    print(f"[generate_answer] synthesizing answer for tool: {next_tool}")

    response = llm.invoke([
        SystemMessage(content=ANSWER_PROMPT),
        HumanMessage(content=f"Question: {question}\n\nData: {context}")
    ])

    answer = response.content.strip()
    print(f"[generate_answer] answer: {answer}")

    return {"final_answer": answer}