from agent.nodes.retrieve_context import FETCH_K, RELATIVE_MARGIN, TOP_K, _collection

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


def _query_topk_unfiltered(query, k=TOP_K):
    r = _collection.query(query_texts=[query], n_results=k, include=["metadatas", "distances"])
    return [(m["db_name"], 1 / (1 + d)) for m, d in zip(r["metadatas"][0], r["distances"][0])]


def _query_topk_relative(query, k=TOP_K, margin=RELATIVE_MARGIN, fetch_k=FETCH_K):
    fetch_k = min(fetch_k, _collection.count())
    r = _collection.query(query_texts=[query], n_results=fetch_k, include=["metadatas", "distances"])
    pairs = [(m["db_name"], 1 / (1 + d)) for m, d in zip(r["metadatas"][0], r["distances"][0])]
    if not pairs:
        return []
    top_score = pairs[0][1]
    kept = [p for p in pairs if p[1] >= top_score - margin][:k]
    return kept


def main():
    print("=" * 70)
    print(f"Unfiltered top-3 vs relative-margin filtered (top-1 +/- {RELATIVE_MARGIN})")
    print("=" * 70)

    changed_count = 0
    for i, query in enumerate(TEST_QUERIES, start=1):
        before = _query_topk_unfiltered(query)
        after = _query_topk_relative(query)

        before_names = [n for n, _ in before]
        after_names = [n for n, _ in after]
        changed = before_names != after_names
        changed_count += changed

        print(f"\n{i}. {query}")
        print("   BEFORE:", [f"{n} ({s:.3f})" for n, s in before])
        print("   AFTER: ", [f"{n} ({s:.3f})" for n, s in after])
        print("   CHANGED:", changed)

    print(f"\n{changed_count} of {len(TEST_QUERIES)} queries changed results with the relative margin applied.")


if __name__ == "__main__":
    main()
