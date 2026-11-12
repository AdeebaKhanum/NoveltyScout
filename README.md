# NoveltyScout - Prior-Art & Patentability Screener (Agentic AI, Assignment 6)

A multi-agent assistant: you describe an invention, agents search papers (arXiv, OpenAlex) and a
vector database of patents/papers, compare your idea feature-by-feature, find novelty gaps and write a
preliminary prior-art note. It is a screening aid, NOT a legal opinion.

## Setup (Windows, Command Prompt) - Python 3.10 to 3.12 recommended
```
cd noveltyscout
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Open `.env` and paste your Gemini key after `GOOGLE_API_KEY=` (or paste it in the app sidebar).

## Run
```
python tests\test_logic.py        (offline sanity tests, optional)
streamlit run app.py
```
First run downloads the small embedding model used by ChromaDB (~80 MB) - needs internet once.

## Use a local model instead of an API
Install Ollama, run `ollama pull llama3.1`, then choose provider **ollama** in the sidebar.

## Folder map
| File | Role |
|---|---|
| app.py | Streamlit GUI (chat, evidence, analysis, report, prompts tabs) |
| noveltyscout/config.py | all settings |
| noveltyscout/llm.py | LLM factory (Gemini / Ollama) + JSON helpers |
| noveltyscout/prompts.py | every system + user prompt |
| noveltyscout/agents.py | router, planner, search (ReAct), novelty, gap (reflection), report |
| noveltyscout/orchestrator.py | intent routing table + handlers |
| noveltyscout/vectorstore.py | ChromaDB wrapper |
| noveltyscout/sources.py | arXiv, OpenAlex, sample data, uploads, chunking |
| noveltyscout/scoring.py | deterministic overlap score + risk flags |
| noveltyscout/memory.py | conversation memory |
| data/sample_prior_art.json | 14 illustrative demo records (not real patents) |
