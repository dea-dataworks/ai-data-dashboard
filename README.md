# AI Data Insight Dashboard — v0.2

**Upload a CSV, explore EDA, run baseline ML, and auto‑generate a client‑ready report.**  
Built with Streamlit, scikit‑learn, and a structured LLM report (Ollama by default, OpenAI optional).

![Demo](assets/demo.gif)

---

## ✨ Key Features (v0.2)

- **Dataset Snapshot (EDA)**
  - Rows × columns, dtype counts, memory footprint
  - Missing values table & top high‑cardinality columns
  - Value counts & categorical distributions
  - Outlier checks (quick visual scanning)

- **ML Insights (Classification & Regression)**
  - Baselines + sensible defaults (Dummy, Logistic Regression, Random Forest)
  - Optional 5‑fold CV; summary and advanced metrics tables
  - **Diagnostics:** Classification — Confusion Matrix, ROC Curve; Regression — Residuals vs Fitted, Prediction Error
  - **Exports:** metrics to CSV/Excel; plots as PNG
  - **Feature Importances:** Random Forest top‑k bar; "Exclude columns" control

- **LLM Report (Tab)**
  - Clean, structured Markdown: snapshot, data quality notes, (optional) key patterns, **modeling summary**, and **feature drivers**
  - Uses **Ollama (Mistral)** by default; optional **OpenAI** provider toggle
  - Handles edge cases (no target/no models) with helpful guidance

- **Nice touches**
  - Sidebar tooltips; compact vs standard plot sizes
  - Consistent captions & explanations for metrics and diagnostics
  - Sample datasets included (Titanic, Insurance) for instant demo

---

## 🚀 Quick Start

> Tested with **Python 3.12–3.13** on Windows/macOS/Linux.

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

### LLM Provider (optional)
- **Default:** Ollama (local) using **Mistral**.
- **OpenAI (cloud):** If available, enable provider toggle in the sidebar (requires `langchain_openai` + `OPENAI_API_KEY`).
- The app safely falls back to Ollama if OpenAI isn’t configured.

---

## 📦 Project Structure (typical)

```
<repo-root>/
├─ app.py
├─ src/
│  ├─ eda.py
│  ├─ ml_models.py
│  ├─ llm_report.py
│  ├─ utils.py
│  └─ __init__.py
├─ data/
│  ├─ titanic.csv
│  └─ insurance.csv
├─ assets/
│  ├─ demo.gif
│  └─ screenshots/
│     ├─ eda.png
│     ├─ ml.png
│     └─ report.png
├─ requirements.txt
├─ README.md
├─ LICENSE
├─ CHANGELOG.md
└─ .gitignore
```
> Adjust as needed to match your repo.

---

## 🖼️ Screenshots

| EDA Snapshot | ML Insights | LLM Report |
| --- | --- | --- |
| ![EDA](assets/screenshots/eda.png) | ![ML](assets/screenshots/ml.png) | ![Report](assets/screenshots/report.png) |

> Optional extras: value counts, outlier plots, feature importance chart.

---

## 🎬 Demo GIF (≤45s)

**Recommended flow:**  
1) Upload `titanic.csv` → 2) Show EDA snapshot → 3) Pick target & run models → 4) Open LLM Report.  
Keep it under **45s** and ≤10–12 MB for GitHub friendliness.

**Tips**
- Use a clean, readable theme (consistent font/plot size).
- Keep the mouse movement slow and purposeful.
- Trim with your editor (or `ffmpeg`) and export to GIF.
- Save to `assets/demo.gif` and ensure relative link in README is correct.

---

## 🛠️ Notes & Configuration

- **Random seed:** For reproducible results in ML tab.
- **CV folds:** Optional 5‑fold cross‑validation for stable metrics.
- **Exports:** Metrics tables → CSV/Excel; plots → PNG (aligned dpi).
- **Excluded columns:** RF importances ignore columns you mark as excluded; LLM report will note exclusions.

---

## 🗺️ Roadmap

- **v0.3 (ideas):**
  - Polished “Key patterns & signals” (with robust numeric/categorical handling)
  - More models (e.g., XGBoost optional), hyperparameter presets
  - Theming polish across plots (consistent sizes/labels/tooltips)
  - Hosted demo link (Streamlit Community Cloud)

See `ROADMAP.md` for the live plan.

---

## 📄 License

This project is released under the **MIT License**. See `LICENSE` for details.

---

## 🙌 Acknowledgments

- Titanic: Kaggle open dataset (trimmed sample).
- Insurance: public sample dataset (charges vs features).

---

## 🔗 Links

- Project Page: <link to repo>
- Author: <your name / website / LinkedIn>
