import json
import re

import chromadb

from agent.state import AgentState

DB_PATH = "data/exercises_db.json"
MATCH_REVIEW_PATH = "match_review.md"
CHROMA_PATH = "data/chroma_db"
COLLECTION_NAME = "exercises"
TOP_K = 3
RELATIVE_MARGIN = 0.08  # rank 2/3 kept only if within this much of the top-1 similarity
FETCH_K = 10  # over-fetch before filtering by margin and truncating to TOP_K

_CANDIDATE_RE = re.compile(r"^- \[X\]\s+(.+?)\s+\(score: [\d.]+\)$", re.IGNORECASE)

_client = chromadb.PersistentClient(path=CHROMA_PATH)


def _parse_confirmed_matches(path=MATCH_REVIEW_PATH):
    """Return {canonical_name: [db_exercise_name, ...]} for [X]-checked candidates only."""
    matches = {}
    current = None
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("## "):
                current = line[3:].strip()
                if current == "Needs manual entry":
                    current = None
                continue
            if current is None:
                continue
            m = _CANDIDATE_RE.match(line)
            if m:
                matches.setdefault(current, []).append(m.group(1))
    return matches


def _build_document(entry):
    parts = [entry.get("name", "")]
    primary = ", ".join(entry.get("primaryMuscles", []))
    secondary = ", ".join(entry.get("secondaryMuscles", []))
    instructions = " ".join(entry.get("instructions", []))
    if primary:
        parts.append(f"Primary muscles: {primary}.")
    if secondary:
        parts.append(f"Secondary muscles: {secondary}.")
    if instructions:
        parts.append(f"Instructions: {instructions}")
    return " ".join(parts)


def _get_or_build_collection():
    collection = _client.get_or_create_collection(COLLECTION_NAME)
    if collection.count() > 0:
        return collection

    with open(DB_PATH, encoding="utf-8") as f:
        db_entries = {entry["name"]: entry for entry in json.load(f)}

    confirmed = _parse_confirmed_matches()

    ids, documents, metadatas = [], [], []
    for canonical_name, db_names in confirmed.items():
        for db_name in db_names:
            entry = db_entries.get(db_name)
            if entry is None:
                continue
            ids.append(entry["id"])
            documents.append(_build_document(entry))
            metadatas.append({"canonical_name": canonical_name, "db_name": db_name})

    if documents:
        collection.add(ids=ids, documents=documents, metadatas=metadatas)
        print(f"[retrieve_context] built collection with {len(documents)} documents")

    return collection


_collection = _get_or_build_collection()


def retrieve_context(state: AgentState) -> dict:
    question = state["messages"][-1].content
    print(f"[retrieve_context] querying: {question}")

    fetch_k = min(FETCH_K, _collection.count())
    results = _collection.query(
        query_texts=[question], n_results=fetch_k, include=["documents", "distances", "metadatas"]
    )
    documents = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    similarities = [1 / (1 + dist) for dist in distances]

    top_docs, top_names = [], []
    if similarities:
        top_score = similarities[0]
        for doc, score, meta in zip(documents, similarities, metadatas):
            if score >= top_score - RELATIVE_MARGIN:
                top_docs.append(doc)
                top_names.append(meta.get("db_name", "Unknown"))
            if len(top_docs) == TOP_K:
                break

    print(f"[retrieve_context] retrieved {len(top_docs)} matches (top-1 + within {RELATIVE_MARGIN} of top score)")
    return {"retrieved_context": top_docs, "retrieved_doc_names": top_names}
