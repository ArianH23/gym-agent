import json
from collections import Counter

from agent.nodes.retrieve_context import DB_PATH, RELATIVE_MARGIN, _collection

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


def part1_doc_counts_per_exercise():
    print("=" * 70)
    print("PART 1: documents per confirmed exercise in the collection")
    print("=" * 70)

    all_data = _collection.get(include=["metadatas"])
    counts = Counter(m["canonical_name"] for m in all_data["metadatas"])

    for name, count in sorted(counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {count:2d}  {name}")
    print(f"\nTotal canonical exercises with confirmed matches: {len(counts)}")
    print(f"Exercises with only 1 confirmed document: {sum(1 for c in counts.values() if c == 1)}")
    print()


def _query_topk_unfiltered(query, k=3):
    r = _collection.query(query_texts=[query], n_results=k, include=["metadatas", "distances"])
    return [
        (m["db_name"], 1 / (1 + d)) for m, d in zip(r["metadatas"][0], r["distances"][0])
    ]


def _query_topk_relative(query, k=3, margin=RELATIVE_MARGIN, fetch_k=10):
    fetch_k = min(fetch_k, _collection.count())
    r = _collection.query(query_texts=[query], n_results=fetch_k, include=["metadatas", "distances"])
    pairs = [
        (m["db_name"], 1 / (1 + d)) for m, d in zip(r["metadatas"][0], r["distances"][0])
    ]
    if not pairs:
        return []
    top_score = pairs[0][1]
    return [p for p in pairs if p[1] >= top_score - margin][:k]


def part2_threshold_comparison():
    print("=" * 70)
    print(f"PART 2: unfiltered top-3 vs relative-margin filtered (top-1 +/- {RELATIVE_MARGIN}) top-3")
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
        print("   AFTER: ", [f"{n} ({s:.3f})" for n, s in after] or "(none within margin)")
        print("   CHANGED:", changed)

    print(f"\n{changed_count} of {len(TEST_QUERIES)} queries changed results with the relative margin applied.")
    print()


def part3_raw_muscle_fields():
    print("=" * 70)
    print("PART 3: raw primaryMuscles / secondaryMuscles for query 1 and query 3 matches")
    print("(unfiltered top-3 — matches what generate_answer actually saw in the prior eval)")
    print("=" * 70)

    with open(DB_PATH, encoding="utf-8") as f:
        db_entries = {e["name"]: e for e in json.load(f)}

    for idx in (1, 3):
        query = TEST_QUERIES[idx - 1]
        matches = _query_topk_unfiltered(query)
        print(f"\nQuery {idx}: {query}")
        if not matches:
            print("  (no matches)")
            continue
        for db_name, score in matches:
            entry = db_entries.get(db_name, {})
            print(f"  - {db_name} (similarity: {score:.3f})")
            print(f"      primaryMuscles:   {entry.get('primaryMuscles')}")
            print(f"      secondaryMuscles: {entry.get('secondaryMuscles')}")
    print()


if __name__ == "__main__":
    part1_doc_counts_per_exercise()
    part2_threshold_comparison()
    part3_raw_muscle_fields()
