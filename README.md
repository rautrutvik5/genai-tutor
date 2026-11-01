# OE GenAI Tutor (RAG)
## Setup
1. Install Ollama (https://ollama.com), then: `ollama pull llama3.2:1b` and `ollama pull nomic-embed-text`
2. `pip install -r requirements.txt`
3. `cp .env.example .env`
4. Handbook already at `data/handbook.pdf` (replace with your own notes if you like)
5. Build index: `python rag.py data/handbook.pdf`
6. Run app: `streamlit run app.py`
7. Chunk/top-k table: `python chunk_experiment.py`
8. Evaluate: `python eval.py` (then fill Y/N columns in results.csv)

## Demo flows
1. Normal question with sources shown
2. Same question with RAG toggle ON vs OFF
3. Out-of-notes question -> "Not found in the notes."

## Roles
A rag.py | B prompts.py | C app.py | D eval.py, questions.csv, REPORT.md
No API keys used; everything runs locally via Ollama.
