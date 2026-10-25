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


def test_combo_context_labels_each_tool_block():
    context = build_context(
        "predict_calories+retrieve_context",
        predicted_calories=500.0,
        prediction_inputs={"Age": 30},
        retrieved_context=["Bench Press targets chest."],
    )
    assert "[predict_calories] 500.00 calories" in context
    assert "[retrieve_context] Bench Press targets chest." in context


def test_combo_context_reports_failed_tool_without_its_data():
    context = build_context(
        "predict_calories+retrieve_context",
        retrieved_context=["Bench Press targets chest."],
        failed_tools={"predict_calories": "network timeout"},
    )
    assert "[predict_calories] unavailable — network timeout" in context
    assert "Do not guess or fabricate" in context
    assert "[retrieve_context] Bench Press targets chest." in context


def test_combo_context_handles_empty_retrieved_context():
    context = build_context(
        "predict_calories+retrieve_context",
        predicted_calories=300.0,
        prediction_inputs={"Age": 25},
        retrieved_context=[],
    )
    assert "[retrieve_context] No matching exercise information was found." in context
