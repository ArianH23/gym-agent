def build_context(next_tool, dataset_results=None, predicted_calories=None,
                   prediction_inputs=None, retrieved_context=None):
    if next_tool == "query_dataset":
        return f"Dataset query result: {dataset_results}"
    if next_tool == "predict_calories":
        return (
            f"Calorie prediction result: {predicted_calories:.2f} calories. "
            f"Based on inputs: {prediction_inputs}"
        )
    if next_tool == "retrieve_context":
        return f"Retrieved context: {' '.join(retrieved_context)}"
    return "No data available."
