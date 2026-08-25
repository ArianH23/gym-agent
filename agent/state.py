from typing import Annotated, TypedDict, List, Dict, Any, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    # 1. Conversational Memory & LLM Calls
    # The add_messages reducer ensures new messages are appended, not overwritten.
    messages: Annotated[List[AnyMessage], add_messages]

    next_tool: Optional[str]  # "query_dataset" | "retrieve_context" | "predict_calories"

    # 2. Routing & Human-in-the-Loop (HITL)
    ambiguous_query: bool  # Set to True by the router/extractor if intent is unclear
    human_feedback: Optional[str]  # Feedback injected by Streamlit UI when graph is resumed

    # 3. Entity Extraction & Semantic Context
    extracted_entities: Dict[str, Any]  # e.g., {"target_age": 30, "workout": "HIIT"}
    retrieved_context: List[str]  # Documents retrieved from ChromaDB
    retrieved_doc_names: List[str]  # Exercise names for the retrieved documents (for UI display)

    # 4. Data Querying (Pandas/CSV)
    pandas_query_code: Optional[str]  # The generated Python/Pandas code
    dataset_results: Optional[str]  # The stringified output of the dataset query

    # 5. ML Model Integration
    prediction_inputs: Dict[str, Any]  # Validated features ready for the regression model
    predicted_calories: Optional[float]  # The numerical output from the MLflow model

    # 6. Final Output
    final_answer: Optional[str]   # The synthesized natural language response
