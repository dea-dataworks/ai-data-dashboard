# Dashboard Walkthrough

## Overview
The dashboard is a Streamlit app for quick EDA, lightweight modeling, and LLM-based report generation.  
This document traces the main execution flow for rehearsal purposes.

---

### app.py Overview
1. Handles layout and tab orchestration.
2. Calls `eda.py`, `ml_models.py`, and `llm_report.py` as main workhorses.
3. Manages session state and caching between tabs.
4. Entry point for the entire data → insight flow.

---

### High-Level Flow
1. User uploads CSV/XLSX → stored in `st.session_state`.
2. Tabs load (`Upload & Preview`, `EDA`, `ML Insights`, `LLM Report`).
3. EDA tab runs summary and visualization functions from `eda.py`.
4. ML tab runs training/evaluation via `ml_models.py` + helpers in `utils.py`.
5. LLM tab reads metrics from cache and generates text summary through `llm_report.py`.
6. Session state maintains results and settings between tabs.

---

### Notes
- Caching prevents redundant training runs.  
- LLM Report depends on ML metrics stored in session state.  

