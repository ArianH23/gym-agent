from agent.context_builder import build_context


def test_query_dataset_context_includes_dataset_results():
    context = build_context("query_dataset", dataset_results="Average calories: 1200")
    assert context == "Dataset query result: Average calories: 1200"


def test_predict_calories_context_includes_formatted_prediction_and_inputs():
    context = build_context(
        "predict_calories",
        predicted_calories=812.5,
        prediction_inputs={"Age": 30, "Gender": "Male"},
    )
    assert "812.50 calories" in context
    assert "'Age': 30" in context


def test_retrieve_context_joins_retrieved_documents():
    context = build_context(
        "retrieve_context",
        retrieved_context=["Split Squats primary muscles: hamstrings.", "Another doc."],
    )
    assert context == "Retrieved context: Split Squats primary muscles: hamstrings. Another doc."


def test_unknown_tool_falls_back_to_no_data_message():
    context = build_context(None)
    assert context == "No data available."


def test_retrieve_context_with_no_documents_produces_empty_join():
    context = build_context("retrieve_context", retrieved_context=[])
    assert context == "Retrieved context: "
