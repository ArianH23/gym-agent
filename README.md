# Fitness RAG Agent

A multi-tool agentic system built with LangGraph that answers natural language questions about a 20,000-row fitness dataset.

## Architecture

- **Router** — LLM-based intent classifier (Gemini) routing queries to the appropriate tool
- **query_dataset** — generates and executes pandas code against the dataset to answer statistical questions
- **predict_calories** — extracts user features from natural language and runs a trained GradientBoosting regression model
- **generate_answer** — synthesizes tool outputs into natural language responses via Gemini

## Stack

LangGraph, LangChain, Gemini, scikit-learn, pandas, ChromaDB (planned), MLflow (planned), Streamlit (planned)

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