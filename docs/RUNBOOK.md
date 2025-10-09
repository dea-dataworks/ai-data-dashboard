# RUNBOOK — AI Data Dashboard v0.2
_Last verified: 2025-10-09_

## 1️⃣ Environment & Config
- Python 3.12 (tested also on 3.13)
- OS: Windows 10
- Virtual env: `.venv` via `python -m venv .venv`
- Dependencies installed via: `pip install -r requirements.txt`
- No secrets required (local Ollama default)
- `.env` added to `.gitignore`
- `.env.example` provided for optional future keys (e.g., `OPENAI_API_KEY`)

## 2️⃣ Run Path & Inputs
- Entry command: `streamlit run app.py`
- Inputs: `data/titanic.csv` or `data/insurance.csv` (or user uploads)
- Working directory: run from project root
- Expected output: Streamlit opens at http://localhost:8501

## 3️⃣ Smoke Test
| Step | Command | Expected Result |
|------|----------|-----------------|
| Launch | `streamlit run app.py` | App opens at http://localhost:8501 |
| Load sample data | Choose Titanic | Table + plots render |
| ML tab | Run baseline models | Metrics tables appear |
| LLM report | Run with Ollama | Markdown summary appears |
All passed on fresh install.

## 4️⃣ Versions Snapshot
Captured via `pip freeze` (top libs only):

```
streamlit==1.38.0
pandas==2.2.2
scikit-learn==1.5.1
langchain==0.2.12
langchain-ollama==0.1.1
ollama==0.2.1
```

## 5️⃣ Notes / Known Quirks
- Ollama must be running (`ollama serve`) before using the LLM Report tab.
- Streamlit cache may hold file locks → restart if CSV upload fails.
- Optional OpenAI provider available (see README for configuration).
- Font fallback: if "Inter" not found, Streamlit uses DejaVu Sans.

## 6️⃣ Maintenance Log
| Date | Action | Result |
|------|---------|--------|
| 2025-10-09 | Fresh venv test | ✅ Pass |
| _Next_ | (update only if env or key dep changes) | |
