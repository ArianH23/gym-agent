import json

import pandas as pd
from rapidfuzz import fuzz, process

CSV_PATH = "data/raw/raw_data.csv"
DB_PATH = "data/exercises_db.json"
OUTPUT_PATH = "match_review.md"
SCORE_THRESHOLD = 90
TOP_N = 5

DUPLICATE_MERGE = {
    "Deadlifts": "Deadlift",
    "Push Ups": "Push-ups",
}


def load_canonical_names():
    df = pd.read_csv(CSV_PATH)

    print("=== COLUMN NAMES ===")
    for col in df.columns:
        print(col)

    unique_names = df["Name of Exercise"].dropna().unique()
    print(f"\n=== UNIQUE VALUES: Name of Exercise ({len(unique_names)}) ===")
    for name in sorted(unique_names):
        print(name)

    canonical = {DUPLICATE_MERGE.get(name, name) for name in unique_names}
    print(f"\n=== CANONICAL NAMES after merge ({len(canonical)}) ===")
    for name in sorted(canonical):
        print(name)

    return sorted(canonical)


def load_db_names():
    with open(DB_PATH, encoding="utf-8") as f:
        db = json.load(f)
    return [entry["name"] for entry in db]


def build_match_review(canonical_names, db_names):
    matched_sections = []
    needs_manual = []

    for name in canonical_names:
        candidates = process.extract(
            name, db_names, scorer=fuzz.WRatio, limit=len(db_names)
        )
        above_threshold = sorted(
            [(cand, score) for cand, score, _ in candidates if score >= SCORE_THRESHOLD],
            key=lambda x: x[1],
            reverse=True,
        )[:TOP_N]
        if above_threshold:
            matched_sections.append((name, above_threshold))
        else:
            needs_manual.append(name)

    return matched_sections, needs_manual


def write_markdown(matched_sections, needs_manual):
    lines = ["# Exercise Match Review", ""]

    for name, candidates in matched_sections:
        lines.append(f"## {name}")
        for cand, score in candidates:
            lines.append(f"- [ ] {cand} (score: {score:.0f})")
        lines.append("")

    lines.append("## Needs manual entry")
    if needs_manual:
        for name in needs_manual:
            lines.append(f"- {name}")
    else:
        lines.append("(none)")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def main():
    canonical_names = load_canonical_names()
    db_names = load_db_names()
    matched_sections, needs_manual = build_match_review(canonical_names, db_names)
    write_markdown(matched_sections, needs_manual)

    print("\n=== SUMMARY ===")
    print(f"Total canonical exercises: {len(canonical_names)}")
    print(f"Matched (>={SCORE_THRESHOLD}): {len(matched_sections)}")
    print(f"Needs manual entry: {len(needs_manual)}")
    print(f"Written to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
