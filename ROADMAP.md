# Roadmap

## v0.2 (Freelance-ready)
### Phase 0 — Repo sanity ✅
- Move modules to `src/`, add `__init__.py`, update `.gitignore`.

### Phase 1 — EDA Schema Panel 
- Snapshot: numeric/categorical/datetime/bool counts, memory, % missing overall.
- Top‑5 high‑cardinality text columns.

### Phase 2 — Outliers & quick relationships
- Boxplot for numeric; value counts (top‑N) for categorical.

### Phase 3 — Data quality warnings
- Duplicates count; leakage checks (feature == target; |corr|≥0.95 numeric).

### Phase 4 — ML baselines + visuals
- DummyClassifier/Regressor; confusion matrix (clf); residuals & y vs ŷ (reg).

### Phase 5 — Optional K‑fold CV
- 5‑fold mean±std for main metrics; toggle in UI.

### Phase 6 — Feature importance
- RF built‑in + permutation importances; top‑10 bar; pass to report.

### Phase 7 — Exports
- Metrics CSV/Excel; plots PNG download.

### Phase 8 — LLM report polish
- Include importances summary; (stretch) provider toggle.

### Phase 9 — README polish + release
- Polish visuals (consistent colors, layout, font sizes)
- GIFs, screenshots, client value section; tag v0.2.0.
