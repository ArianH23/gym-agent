from langchain_core.messages import HumanMessage

from agent.graph import graph

result = graph.invoke({
    "messages": [HumanMessage(content="What muscles does a Bulgarian Split Squat target?")],
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

print("\n--- FINAL STATE ---")
print("Tool used:", result["next_tool"])
print("Retrieved context:", result["retrieved_context"])
print("Final answer:", result["final_answer"])
