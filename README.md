# Fitness RAG Agent

A multi-tool agentic system built with LangGraph that answers natural language questions about a 20,000-row fitness dataset.

## Architecture

- **Router** — LLM-based intent classifier (Gemini) routing queries to the appropriate tool
- **query_dataset** — generates and executes pandas code against the dataset to answer statistical questions
- **predict_calories** — extracts user features from natural language and runs a trained GradientBoosting regression model
- **retrieve_context** — semantic search over a ChromaDB collection of exercise descriptions (name + muscles + instructions), sourced from [free-exercise-db](https://github.com/yuhonas/free-exercise-db) and matched against the project's exercise list via fuzzy matching (see `match_review.md`)
- **generate_answer** — synthesizes tool outputs into natural language responses via Gemini

## Stack

LangGraph, LangChain, Gemini, scikit-learn, pandas, ChromaDB, MLflow (planned), Streamlit (planned)

## Known Limitations

1. **Retrieval quality depends on document density per exercise.** Of the 21 confirmed exercises in the ChromaDB collection, 12 have only 1 matched document. Any top-3 retrieval for those exercises structurally has to pull in unrelated filler to fill the remaining slots, regardless of embedding model — there's simply nothing else on-topic to retrieve.
2. **Gemini embeddings were evaluated as an alternative to ChromaDB's default (MiniLM) and reverted.** Swapping to Gemini's `gemini-embedding-001` did not improve semantic discrimination in this narrow, vocabulary-overlapping domain. It also compressed similarity scores into a tight band (~0.84–0.90 regardless of relevance) versus MiniLM's wider spread (~0.42–0.70), making similarity scores a less useful confidence signal. The default MiniLM embedding function is what's currently in use.

   The underlying discrimination weakness is a recurring pattern in this domain, not a one-off — confirmed with two separate documented examples:
   - **"What muscles does a Pull-up work?"** — correct top-1 (Pullups, 0.568), but ranks 2–3 are unrelated (Face Pull 0.562, Incline Push-Up/Pushups ~0.55–0.56), with too small a score gap for any threshold to separate signal from noise.
   - **"What muscles do Lateral Raises target?"** — worse: the correct document ("Seated Side Lateral Raise") isn't even in the top 3. It ranks **14th of 48** at similarity 0.491, below three unrelated exercises (Cable Shoulder Press, Incline Push-Up, Rear Leg Raises, all 0.51–0.52). The right answer exists in the collection but the embedding model fails to surface it at all.

   Trying alternate candidate variants for Lateral Raises to work around this was considered and rejected — that would mask the underlying embedding weakness rather than fix it.
3. **The source data (free-exercise-db) has inconsistent muscle attribution across variants of the same base exercise.** For example, different Bench Press variants disagree on whether the primary muscle is chest or triceps depending on equipment. This is a source-data limitation, not a retrieval or synthesis bug — `generate_answer` reports what's in the retrieved documents faithfully.

Further embedding-function or similarity-threshold tuning is not planned. The next priority is writing the 26 manual entries for currently-unconfirmed exercises (see `match_review.md`), which addresses the root cause (missing data) rather than compensating for it at retrieval time.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add your Google API key to `.env`:

```
GOOGLE_API_KEY=your_key_here
```

Run:
```bash
python run_test.py
```