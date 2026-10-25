def build_context(next_tool, dataset_results=None, predicted_calories=None,
                   prediction_inputs=None, retrieved_context=None, failed_tools=None):
    if next_tool and "+" in next_tool:
        return _build_combo_context(
            next_tool.split("+"),
            dataset_results=dataset_results,
            predicted_calories=predicted_calories,
            prediction_inputs=prediction_inputs,
            retrieved_context=retrieved_context,
            failed_tools=failed_tools or {},
        )

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


def _build_combo_context(tools, dataset_results, predicted_calories,
                          prediction_inputs, retrieved_context, failed_tools):
    blocks = []
    for tool in tools:
        if tool in failed_tools:
            blocks.append(
                f"[{tool}] unavailable — {failed_tools[tool]}. "
                "Do not guess or fabricate this part of the answer."
            )
        elif tool == "query_dataset":
            blocks.append(f"[query_dataset] {dataset_results}")
        elif tool == "predict_calories":
            if predicted_calories is None:
                blocks.append("[predict_calories] No prediction was produced.")
            else:
                blocks.append(
                    f"[predict_calories] {predicted_calories:.2f} calories, "
                    f"based on inputs: {prediction_inputs}"
                )
        elif tool == "retrieve_context":
            if retrieved_context:
                blocks.append(f"[retrieve_context] {' '.join(retrieved_context)}")
            else:
                blocks.append("[retrieve_context] No matching exercise information was found.")
        else:
            blocks.append(f"[{tool}] Unknown tool.")
    return "\n".join(blocks)
