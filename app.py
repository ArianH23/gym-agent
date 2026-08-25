import streamlit as st
from langchain_core.messages import HumanMessage

from agent.graph import graph

EXAMPLE_QUESTIONS = [
    "What is the average calories burned by female users?",
    "Predict calories burned for a 30 year old male, 80kg, 1.75m doing 45 minutes of HIIT, 3 times a week, intermediate level",
    "What muscles does a Bulgarian Split Squat target?",
    "What's the best way to get fit?",
]

st.set_page_config(page_title="Fitness Agent", page_icon="\U0001F3CB")
st.title("Fitness Agent")

if "history" not in st.session_state:
    st.session_state.history = []
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

with st.sidebar:
    st.subheader("Example questions")
    for example in EXAMPLE_QUESTIONS:
        if st.button(example, use_container_width=True):
            st.session_state.query_input = example
            st.rerun()


def run_query(question: str) -> dict:
    return graph.invoke({
        "messages": [HumanMessage(content=question)],
        "ambiguous_query": False,
        "human_feedback": None,
        "extracted_entities": {},
        "retrieved_context": [],
        "retrieved_doc_names": [],
        "pandas_query_code": None,
        "dataset_results": None,
        "prediction_inputs": {},
        "predicted_calories": None,
        "final_answer": None,
        "next_tool": None,
    })


question = st.text_input("Ask a question", key="query_input")
submitted = st.button("Submit")

if submitted and question.strip():
    try:
        result = run_query(question)
        st.session_state.history.append({
            "question": question,
            "answer": result.get("final_answer"),
            "tool": result.get("next_tool"),
            "doc_names": result.get("retrieved_doc_names") or [],
            "error": None,
        })
    except Exception as e:
        st.session_state.history.append({
            "question": question,
            "answer": None,
            "tool": None,
            "doc_names": [],
            "error": str(e),
        })

st.divider()

for entry in reversed(st.session_state.history):
    st.markdown(f"**You:** {entry['question']}")
    if entry["error"]:
        st.error(f"Something went wrong answering this question: {entry['error']}")
    else:
        st.markdown(f"**Agent:** {entry['answer']}")
        if entry["tool"]:
            st.caption(f"Handled by: {entry['tool']}")
        if entry["tool"] == "retrieve_context" and entry["doc_names"]:
            st.caption("Retrieved documents: " + ", ".join(entry["doc_names"]))
    st.divider()
