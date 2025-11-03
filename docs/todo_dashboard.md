# Dashboard — v0.3 TODOs

This file tracks pending improvements and refactors planned for the next version (v0.3: Docker + FastAPI).

---

## Core Refactors
- [ ] **Modularize ML tab (Tab 3)**  
  Split the long `tab3` logic in `app.py` into smaller functions within `ml_models.py`  
  (UI config, training logic, metrics rendering, plots).

- [ ] **Simplify CV / single-split branching**  
  Combine both modeling paths into one helper that handles validation mode internally.

- [ ] **Centralize exception handling**  
  Create `utils.handle_error()` to log errors to file and sidebar, replacing inline `except` blocks.

- [ ] **Unify feature-importance extraction**  
  Move Random Forest importance extraction into a single helper function usable by both tasks.

---

## Infrastructure & Integration
- [ ] **Add Docker support**  
  Write a `Dockerfile` for reproducible local runs; test image build and container launch.

- [ ] **Add FastAPI wrapper**  
  Create a lightweight API (`api.py`) exposing EDA summary and model predictions.

- [ ] **Set up basic CI/CD**  
  GitHub Actions workflow: lint, test, build Docker image on push.

---

## UI & Usability
- [ ] **Improve section labeling in ML tab**  
  Update Streamlit headers for clarity (e.g., “Classification Models,” “Regression Models”).

- [ ] **Standardize plot sizing & layout**  
  Ensure consistent dimensions and captions across EDA and ML tabs.

- [ ] **Optional: add sidebar log viewer**  
  Surface key `logging` output interactively for debugging and user feedback.

---

## Notes
These tasks prepare the app for v0.3 “productionization” — containerized, modular, and ready for API integration.
