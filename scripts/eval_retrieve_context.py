from langchain_core.messages import HumanMessage

from agent.graph import graph
from agent.nodes.retrieve_context import TOP_K, _collection

import mlflow

OUTPUT_PATH = "eval_results.md"

TEST_QUERIES = [
    "What muscles does a Bulgarian Split Squat target?",
    "How do I do a proper Deadlift?",
    "What's the difference between a compound and isolation exercise, using Bench Press as an example?",
    "What muscles does a Pull-up work?",
    "How should I do a Plank correctly?",
    "What's the proper form for a Squat?",
    "What muscles does a Bicep Curl target?",
    "What does a Lat Pulldown work?",
]


def run_eval():
    sections = ["# retrieve_context Evaluation", ""]

    for i, query in enumerate(TEST_QUERIES, start=1):
        with mlflow.start_run():
            mlflow.log_param("query", query)
            print(f"[{i}/{len(TEST_QUERIES)}] {query}")

            raw = _collection.query(
                query_texts=[query],
                n_results=TOP_K,
                include=["metadatas", "distances"],
            )
            metadatas = raw["metadatas"][0]
            distances = raw["distances"][0]

            result = graph.invoke({
                "messages": [HumanMessage(content=query)],
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

            sections.append(f"## {i}. {query}")
            sections.append("")
            sections.append("**Top 3 retrieved documents:**")
            sections.append("")
            sections.append("| Rank | Document name | Distance (lower = closer) | Similarity (1 / (1 + distance)) |")
            sections.append("|---|---|---|---|")
            for rank, (meta, dist) in enumerate(zip(metadatas, distances), start=1):
                mlflow.log_metric(f"distance_rank_{rank}", dist)

                similarity = 1 / (1 + dist)
                sections.append(
                    f"| {rank} | {meta['db_name']} (matched to \"{meta['canonical_name']}\") | {dist:.4f} | {similarity:.4f} |"
                )
            sections.append("")
            sections.append(f"**Final synthesized answer:** {result['final_answer']}")
            sections.append("")

            with open(f"{OUTPUT_PATH}_ml.txt", "w") as f:
                f.write(result['final_answer'])

            mlflow.log_artifact(f"{OUTPUT_PATH}_ml.txt")


    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(sections) + "\n")


    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    run_eval()
