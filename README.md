# AI Data Dashboard (Advanced)

This repository contains feature-rich version of the AI Data Dashboard. For a streamlined implementation rebuilt from scratch with a simplified architecture, see the ai-data-dashboard repository.


**Upload a CSV, explore EDA, run baseline ML, and auto‑generate a client‑ready report.**  
Built with Streamlit, scikit‑learn, and a structured LLM report (Ollama by default, OpenAI optional).

![Demo](docs/README_assets/demo.gif)

---

## Key Features (v0.2)

- **Dataset Preview & EDA**
  - Quick dataset snapshot: shape, dtypes, memory, missing values
  - Automatic visualizations: distributions, outlier scans, high-cardinality detection

- **Machine Learning Insights**
  - Built-in models (Dummy, Logistic Regression, Random Forest)
  - One-click metrics & diagnostics: confusion matrix, ROC, residuals, prediction error
  - Optional cross-validation, feature importances, and column exclusion
  - Export results (metrics → CSV/Excel, plots → PNG)

- **LLM-Powered Report**
  - Structured Markdown with dataset snapshot, data quality notes, modeling summary, and feature drivers
  - Runs locally via **Ollama (Mistral)** or optionally via **OpenAI API**

- **Streamlined UX**
  - Sidebar tooltips, compact/standard plot sizing
  - Consistent captions & metric explanations
  - Dark/light theme friendly
  - Sample datasets included for instant demo

---

## Quick Start

> Tested with **Python 3.12–3.13** on Windows.

```bash
# 1) Clone
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>

# 2) Create & activate a venv (example: Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux:
# python3 -m venv .venv
# source .venv/bin/activate

# 3) Install
pip install -r requirements.txt

# 4) Run
streamlit run app.py
```
Then open the local URL that Streamlit prints (usually http://localhost:8501).

### LLM Providers

By default, the dashboard uses **Ollama (local Mistral)**. You can also enable **OpenAI (cloud)** if you have an API key.  

#### Option 1 — Ollama (default, local)  
1. [Install Ollama](https://ollama.ai/download) for your OS.  
2. Pull the Mistral model (once):  
   ```bash
   ollama run mistral
   ```  
   This downloads the model (~4 GB) and verifies it runs locally.  
3. Run the dashboard as normal (`streamlit run app.py`). Ollama will be used automatically.  

#### Option 2 — OpenAI (optional, cloud)  
1. Requirements already include OpenAI support (`langchain-openai`, `openai`).  
2. Set your API key in your environment:  
   ```bash
   export OPENAI_API_KEY=your_key_here   # macOS/Linux
   setx OPENAI_API_KEY "your_key_here"   # Windows (PowerShell)
   ```  
3. Launch the dashboard, open the sidebar, and select **OpenAI** under *LLM Provider*.  
4. If OpenAI isn’t configured, the app safely falls back to Ollama.  

---

## Project Structure 

```
ai-data-dashboard/
├─ .env.example
├─ .gitignore
├─ LICENSE
├─ README.md
├─ app.py
│
├─ docs/
│  ├─ README_assets/
│  │  ├─ demo.gif
│  │  ├─ eda.gif
│  │  ├─ ml.gif
│  │  └─ screenshots/
│  │     ├─ eda.png
│  │     ├─ ml.png
│  │     ├─ preview.png
│  │     └─ report.png
│  │
├─ examples/
│  ├─ insurance.csv
│  └─ titanic.csv
│
├─ requirements.txt
│
├─ src/
│  ├─ __init__.py
│  ├─ eda.py
│  ├─ fonts/
│  │  └─ Inter-VariableFont_opsz,wght.ttf
│  ├─ llm_report.py
│  ├─ ml_models.py
│  └─ utils.py

```

---

## Screenshots

| Preview Snapshot | EDA Snapshot | ML Insights | LLM Report |
| --- | --- | --- | --- |
|![PREVIEW](docs/README_assets/screenshots/preview.png) | ![EDA](docs/README_assets/screenshots/eda.png) | ![ML](docs/README_assets/screenshots/ml.png) | ![Report](docs/README_assets/screenshots/report.png) |

---

### Extra GIFs

- [EDA expanders demo (GIF)](docs/README_assets/eda.gif)  
- [ML expanders demo (GIF)](docs/README_assets/ml.gif)

## Notes & Configuration

- **Random seed:** For reproducible results in ML tab.
- **CV folds:** Optional 5‑fold cross‑validation for stable metrics.
- **Exports:** Metrics tables → CSV/Excel; plots → PNG (aligned dpi).
- **Excluded columns:** RF importances ignore columns you mark as excluded; LLM report will note exclusions.
- It runs without secrets. If you want to use a cloud LLM later, copy `.env.example` to `.env` and fill in    values.  
- The app automatically detects the `.env` file at startup (no extra setup needed).

---

## License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## Acknowledgments

- Titanic: Kaggle open dataset (trimmed sample).
- Medical Cost: Kaggle open dataset (trimmed sample).

---

## 🔗 Links

- Project Page: https://github.com/daniel-e-alarcon/ai-data-dashboard-advanced
- Author: Daniel E. Alarcon/ https://www.linkedin.com/in/daniel-e-alarcon