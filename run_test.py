from langchain_core.messages import HumanMessage
from agent.graph import graph

result = graph.invoke({
    "messages": [HumanMessage(content="What is the average calories burned by female users?")],
    "ambiguous_query": False,
    "human_feedback": None,
    "extracted_entities": {},
    "retrieved_context": [],
    "pandas_query_code": None,
    "dataset_results": None,
    "prediction_inputs": {},
    "predicted_calories": None,
    "final_answer": None,
    "next_tool": None,
})

print("\n--- FINAL STATE ---")
print("Tool used:", result["next_tool"])
print("Extracted features:", result["prediction_inputs"])
print("Predicted calories:", result["predicted_calories"])
