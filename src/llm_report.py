from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
import streamlit as st
import src.utils as utils
from langchain_ollama import OllamaLLM
try:
    from langchain_openai import ChatOpenAI
    _OPENAI_OK = True
except Exception:
    _OPENAI_OK = False

# ---- Safe helper: get OpenAI key without crashing when secrets.toml is missing
def _get_openai_key() -> str | None:
    try:
        # st.secrets raises if no secrets file exists; guard it
        return st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        return os.getenv("OPENAI_API_KEY")

# ---------- Utilities to format data for the prompt ----------

def _top_items(d: Dict[Any, Any], k: int = 10) -> Dict[Any, Any]:
    """Keep only top-k keys by absolute value if numeric; otherwise first k items."""
    if not d:
        return d
    try:
        return dict(sorted(d.items(), key=lambda x: abs(float(x[1])), reverse=True)[:k])
    except Exception:
        # Fallback for non-numeric dicts
        return dict(list(d.items())[:k])

def _fmt_dict(d: Dict[Any, Any], max_items: int = 10) -> str:
    """Pretty bullet list for dictionaries, truncated."""
    if not d:
        return "not available"
    items = list(d.items())[:max_items]
    rows = [f"- {str(k)}: {str(v)}" for k, v in items]
    truncated = len(d) > max_items
    return "\n".join(rows) + ("\n- … (truncated)" if truncated else "")

def _safe_corr_with_target(df: pd.DataFrame, target: str, top_k: int = 10) -> Dict[str, float]:
    """Compute numeric correlations with target; quietly handle edge cases."""
    if target not in df.columns:
        return {}
    num_df = df.select_dtypes(include=[np.number])
    if target not in num_df.columns:
        return {}
    corr_series = num_df.corr(numeric_only=True)[target].drop(labels=[target], errors="ignore")
    corr_series = corr_series.replace([np.inf, -np.inf], np.nan).dropna()
    # strongest absolute correlations first
    corr_series = corr_series.reindex(corr_series.abs().sort_values(ascending=False).index)
    return corr_series.head(top_k).round(3).to_dict()

def _trim_corr_text(corr_dict: Dict[str, float], min_abs: float = 0.10, max_items: int = 8) -> str:
    if not corr_dict:
        return "No strong correlations (|r| ≥ 0.10)."
    strong = [(k, v) for k, v in corr_dict.items() if abs(v) >= min_abs]
    if not strong:
        return "No strong correlations (|r| ≥ 0.10)."
    strong = strong[:max_items]
    return "\n".join(f"- {k}: {v:+.3f}" for k, v in strong)

def _class_balance(df: pd.DataFrame, target: str) -> Dict[Any, int]:
    if target not in df.columns:
        return {}
    counts = df[target].value_counts(dropna=False)
    return counts.to_dict()

def _column_types(df: pd.DataFrame) -> Dict[str, str]:
    return {col: str(dtype) for col, dtype in df.dtypes.items()}

def _missing_values(df: pd.DataFrame) -> Dict[str, int]:
    miss = df.isna().sum()
    miss = miss[miss > 0].sort_values(ascending=False)
    return miss.to_dict()

def _shape_pretty(df: pd.DataFrame) -> str:
    return f"{df.shape[0]} rows × {df.shape[1]} columns"

def _key_vars_human(df: pd.DataFrame, max_vars: int = 7) -> str:
    cols = list(df.columns[:max_vars])
    # Light grouping hint (names only, keeps it generic across datasets)
    return ", ".join(cols)

def _corr_to_sentences(corr_dict: Dict[str, float], target: Optional[str]) -> str:
    if not corr_dict:
        return "No strong correlations (|r| ≥ 0.10)."
    tgt = target or "the target"
    lines = []
    for feat, r in corr_dict.items():
        direction = "positively" if r > 0 else "negatively"
        lines.append(f"- {feat} is {direction} associated with {tgt} (r = {r:+.3f}).")
    return "\n".join(lines)

# ---------- Feature-driver helpers (collapse one-hot, rank, interpret) ----------
def _strip_pipeline_prefix(name: str) -> str:
    for pre in ("num__", "cat__", "text__", "bin__"):
        if name.startswith(pre):
            return name[len(pre):]
    return name

def _collapse_one_hot_importances(fi: Optional[Dict[str, float]]) -> list[tuple[str, float]]:
    """
    Collapse groups like Sex_male/Sex_female -> "Sex (male)" with the group's top suffix by importance.
    Keeps single columns as-is. Returns list of (pretty_name, score) sorted desc by score.
    """
    if not fi:
        return []
    cleaned: Dict[str, float] = {_strip_pipeline_prefix(k): float(v) for k, v in fi.items()}

    # Group by base (text before first underscore). If we see >=2 suffix variants, treat as one-hot group.
    by_base: Dict[str, list[tuple[str, float]]] = {}
    for name, score in cleaned.items():
        base, sep, suffix = name.partition("_")
        if sep:  # looks like one-hot
            by_base.setdefault(base, []).append((suffix, score))
        else:
            by_base.setdefault(base, []).append(("", score))

    aggregated: list[tuple[str, float]] = []
    for base, parts in by_base.items():
        # one-hot if we have at least two suffixed variants
        has_ohe = sum(1 for s, _ in parts if s) >= 2
        if has_ohe:
            best_suffix, best_score = max(parts, key=lambda t: t[1])
            pretty = f"{base} ({best_suffix})"
            aggregated.append((pretty, best_score))
        else:
            # either a single plain column or only one suffixed variant
            sfx, score = parts[0]
            pretty = base if sfx == "" else f"{base} ({sfx})"
            aggregated.append((pretty, score))

    aggregated.sort(key=lambda t: t[1], reverse=True)
    return aggregated

def _interpret_driver(name: str, dataset_name: str) -> str:
    """Tiny interpretation layer; Titanic-aware, generic otherwise."""
    ds = (dataset_name or "").lower()
    base = name.split(" (")[0]
    if "titanic" in ds:
        if name.startswith("Sex"):
            return "males had much lower survival."
        if base == "Pclass":
            return "higher class increased survival likelihood."
        if base == "Age":
            return "younger passengers showed better outcomes."
        if base == "SibSp":
            return "more siblings/spouses generally reduced survival odds."
        if base == "Fare":
            return "higher fares correlated with better survival."
    return "important for predicting the target in this dataset."

def _feature_drivers_markdown(fi: Optional[Dict[str, float]], dataset_name: str, top_k: int = 3) -> str:
    """Build final markdown list (ranked 1..k) ready to print in the report."""
    agg = _collapse_one_hot_importances(fi)
    if not agg:
        return "N/A"
    lines = []
    for i, (pretty, _score) in enumerate(agg[:top_k], start=1):
        lines.append(f"{i}. **{pretty}:** {_interpret_driver(pretty, dataset_name)}")
    return "\n".join(lines) if lines else "N/A"

# ---------- Main function: compute stats, call LLM ----------

def llm_report_tab(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    target: Optional[str] = None,
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    feature_importances: Optional[Dict[str, float]] = None,
    recommendations: Optional[str] = None,
    llm_model_name: str = "mistral",
    models_table_md: Optional[str] = None,
    top_k_corr: int = 10,
    excluded_columns: Optional[list[str]] = None,
    llm=None,
) -> str:
    """
    Build a structured, factual report using a Runnable pipeline (prompt | llm | StrOutputParser).
    Pass in optional model_metrics (per-model dict with accuracy/f1/auc/precision/recall) and feature_importances.
    Returns the report string.
    """

    # ---- Compute facts from DataFrame ----
    shape = tuple(df.shape)
    col_types = _column_types(df)
    sample_columns = list(df.columns[:min(8, len(df.columns))])

    class_balance = _class_balance(df, target) if target else {}
    missing_vals = _missing_values(df)
    use_df = df.drop(columns=excluded_columns or [], errors="ignore")
    correlations = _safe_corr_with_target(use_df, target, top_k=top_k_corr) if target else {}
    correlations_text = _trim_corr_text(correlations, min_abs=0.10, max_items=8)
    correlations_sent = _corr_to_sentences(correlations, target)

    sentences = []
    if correlations:
        for feat, r in list(correlations.items()):
            if abs(r) < 0.10:
                continue
            strength = "slightly " if abs(r) < 0.20 else "moderately " if abs(r) < 0.40 else "strongly "
            direction = "positive" if r > 0 else "negative"
            tgt = target or "the target"
            if abs(r) >= 0.20:
                sentences.append(f"- {feat} shows a {strength}{direction} association with {tgt} (r = {r:+.2f}).")
            else:
                sentences.append(f"- {feat} shows a {strength}{direction} association with {tgt}.")
    correlations_sent = "\n".join(sentences[:3]) if sentences else "No strong correlations (|r| ≥ 0.10)."
       
    if recommendations is None:
        recommendations = ""

    # Model lists / metrics formatting
    model_list: List[str] = list(model_metrics.keys()) if model_metrics else []
    model_metrics_str = _fmt_dict(model_metrics or {}, max_items=20)
    # feature_importances = _top_items(feature_importances or {}, k=15)
    # feature_importances_str = _fmt_dict(feature_importances, max_items=15)

    # # Feature importances: pass only the top-5 names (no numbers) to encourage interpretation
    # fi_dict = _top_items(feature_importances or {}, k=5)
    # def _clean_feat_name(name: str) -> str:
    #     # strip common pipeline prefixes to make names nicer in prose
    #     for pre in ("num__", "cat__", "text__", "bin__"):
    #         if name.startswith(pre):
    #             return name[len(pre):]
    #     return name
    # fi_names = [ _clean_feat_name(n) for n, _ in sorted(fi_dict.items(), key=lambda x: float(x[1]), reverse=True) ]
    # fi_names_str = ", ".join(fi_names)

    # Build finalized, human-friendly Feature Drivers (ranked, top-3, one-hot collapsed)
    feature_drivers_md = _feature_drivers_markdown(feature_importances, dataset_name, top_k=3)
 

    template = """
        You are a precise data analyst. Write a concise, client-ready **Markdown** report.
        **Use only** the values provided in the inputs below; if something is missing, write **N/A**.
        Prefer short paragraphs and tight bullet points over raw dumps. Avoid hedging.

        # Data Analysis Report — {dataset_name}

        ## 1) Dataset Snapshot
        - **Shape:** {shape}
        - **Target:** {target}
        - **Key variables:** {sample_columns}
        - **Excluded columns:** {excluded_note}
        - In one sentence, characterize the dataset at a high level (no long lists).
         If any remaining columns look like IDs or ticket codes, briefly flag them as 
         potentially high-cardinality features of limited predictive value.

       ## 2) Data Quality Notes
        Produce a **single table** followed by at most **2 bullets**. Do **not** use generic language.

        **Table (required):** For each entry in **Missing values (top)** below, create a row with:
        `Column | Missing (n/N, ~%) | Severity | Recommended action`

        Rules (mandatory):
        - Let **N** be the number of rows from **Shape** (e.g., “50 rows × …” → N=50).
        - If n/N ≥ 0.70 → **Severity = Very high** and **Recommended action = Drop (likely unusable)**.
        - If 0.10 ≤ n/N < 0.70 → **Severity = Moderate** and **Recommended action = Impute** (numeric: median or group-based; categorical: mode or group-based).
        - If n/N < 0.10 → **Severity = Low** and **Recommended action = Minor/ignore or simple impute**.
        - Use the exact phrases above for **Severity** and **Recommended action**. Do **not** call >70% “moderate.”

        **Bullets (optional, ≤2):**
        - If any columns *appear* identifier-like or **high-cardinality** (e.g., “Ticket”), add one bullet: “`<col>` appears high-cardinality; limited predictive value unless engineered.”
        - If duplicates or class imbalance are indicated elsewhere, add one bullet with a specific action (deduplicate; use stratified split/metrics beyond accuracy).

        **Missing values (top) input:**  
        {missing_values}

        ## 3) Modeling Summary
        Write one or two sentences comparing model performance **based on the metrics table below**.  
        - Identify which models perform clearly better than the baseline (Dummy Classifier).  
        - If two or more models are competitive, describe the trade-offs (e.g., one stronger on Accuracy/F1, another on ROC AUC).  
        - Use only qualitative terms such as “slightly better,” “comparable,” or “clearly stronger.”  
        - Do not calculate or invent exact percentages or restate each row in prose.  
        Then present the provided table verbatim.
        {models_table_md}

        ## 4) Feature Drivers (ranked by RF importance)
        Rewrite the input below as a **Markdown numbered list (1., 2., 3.)**.  
        Each item must:
        - Bold the feature name,  
        - Keep the short interpretation,  
        - Follow the ranked order exactly.  
        If input says N/A, write N/A.
        {feature_drivers_md}


        ## 5) Practical Takeaways
        Provide **4–6** specific, actionable recommendations tied to the issues and signals above.
        Be concrete (what to impute/drop/engineer, how to validate). If little is available, keep this brief.
        {recommendations}

        ## 6) Notes
        - **Excluded columns:** {excluded_note}
        - If any section lacked inputs, note it as **N/A** and proceed without guessing.
        """


    prompt = PromptTemplate(
        template=template.strip(),
        input_variables=[
            "dataset_name", "shape", "column_types", "sample_columns", "target",
            "class_balance", "missing_values", "correlations",
            "model_list", "model_metrics", "recommendations", "models_table_md",
            "excluded_note", "feature_drivers_md"
        ],
    )

    # ---- Build inputs (as readable bullet lists / strings) ----
    inputs = {
        "dataset_name": dataset_name,
        "models_table_md": models_table_md or "Not run.",
        "shape": _shape_pretty(df),
        "column_types": _fmt_dict(col_types, max_items=50),
        #"sample_columns": _key_vars_human(df),
        "sample_columns": _key_vars_human(df.drop(columns=excluded_columns or [], errors="ignore")),
        "target": target or "not specified",
        "class_balance": _fmt_dict(class_balance, max_items=20) if class_balance else "not available",
        "missing_values": _fmt_dict(missing_vals, max_items=50) if missing_vals else "no missing values",
        "correlations": correlations_sent,
        #"feature_importances": (fi_names_str if fi_names else "not available"),
        "feature_drivers_md": feature_drivers_md,
        "model_list": ", ".join(model_list) if model_list else "not available",
        "model_metrics": model_metrics_str if model_metrics else "not available",
        "recommendations": recommendations,
        "excluded_note": (", ".join(excluded_columns) if excluded_columns else "None"),
     }
    
    # ---- Runnable pipeline: prompt | llm | StrOutputParser ----
    llm = llm or OllamaLLM(model=llm_model_name)
    chain = prompt | llm | StrOutputParser()

    try:
        report: str = chain.invoke(inputs)
        return report
    except Exception as e:
        # Normalize provider errors so the caller can show a clean message
        raise RuntimeError(str(e)) from e

def _format_model_metrics(models_output: dict | None) -> str:
    """
    Build a Markdown table from your models_output dict.
    Supports both classification (accuracy, f1, roc_auc) and regression (r2, mae, rmse).
    Returns '' if nothing to show.
    """
    if not models_output or "results" not in models_output:
        return ""

    results = models_output["results"]
    # Heuristically detect task type from first model’s keys
    sample = next(iter(results.values()))
    is_classification = "accuracy" in sample or "f1_score" in sample or "roc_auc" in sample

    lines = []
    if is_classification:
        lines.append("| Model | Accuracy | F1 (weighted) | ROC AUC |")
        lines.append("|---|---:|---:|---:|")
        for model, m in results.items():
            acc = m.get("accuracy", float("nan"))
            f1  = m.get("f1_score", float("nan"))
            auc = m.get("roc_auc", None)
            auc_str = f"{auc:.3f}" if isinstance(auc, (int, float)) else "—"
            lines.append(f"| {model} | {acc:.3f} | {f1:.3f} | {auc_str} |")
    else:
        lines.append("| Model | R² | MAE | RMSE |")
        lines.append("|---|---:|---:|---:|")
        for model, m in results.items():
            r2  = m.get("r2_score", float("nan"))
            mae = m.get("mae", float("nan"))
            rmse = m.get("rmse", float("nan"))
            lines.append(f"| {model} | {r2:.3f} | {mae:.3f} | {rmse:.3f} |")

    return "\n".join(lines)

# --- Signature diff helpers (explain why cache is invalid) ---
_SIG_FIELDS = ["dataset_name", "shape", "columns", "target", "excluded_cols", "cv_used", "cv_folds", "seed"]

def _diff_signature(old_sig, new_sig):
    """Return list of (field, old, new) for items that differ."""
    diffs = []
    for name, a, b in zip(_SIG_FIELDS, old_sig, new_sig):
        if a != b:
            diffs.append((name, a, b))
    return diffs

def _human_reason(diffs):
    """Turn signature diffs into a short human hint."""
    # Prioritize the most common user-facing causes
    order = ["dataset_name", "target", "excluded_cols", "cv_used", "cv_folds", "seed", "columns", "shape"]
    by_name = {n: (n, a, b) for (n, a, b) in diffs}
    for key in order:
        if key in by_name:
            n, a, b = by_name[key]
            if n == "dataset_name":
                return "Dataset changed"
            if n == "target":
                return f"Target changed (was '{a}', now '{b}')"
            if n == "excluded_cols":
                a_len = len(a or [])
                b_len = len(b or [])
                return f"Excluded columns changed ({a_len} → {b_len})"
            if n == "cv_used":
                return "CV vs single-split setting changed"
            if n == "cv_folds":
                return f"CV folds changed ({a} → {b})"
            if n == "seed":
                return f"Random seed changed ({a} → {b})"
            if n == "columns":
                return "Dataset columns changed"
            if n == "shape":
                return "Dataset shape changed"
    # Fallback if we get here
    return "settings changed"


# --- Cache helper: use ML tab artifacts, do not retrain here ---
def cached_ml_artifacts(df, dataset_name: str, target: str | None):
    """
    Returns (status, models_table_md, rf_importances, context)
    status ∈ {"ok","no_target","no_ml","missing_table","mismatch"}
    context: dict with optional keys like {"reason": "..."} for display.
    """
    if not target:
        return "no_target", "", None, {}

    excluded = st.session_state.get("ml_excluded_cols", [])
    cv_used  = st.session_state.get("ml_cv_used", False)
    cv_folds = st.session_state.get("ml_cv_folds", None)

    current_sig = utils.ml_signature(
        df,
        st.session_state.get("dataset_name", dataset_name), 
        target,
        excluded,
        cv_used,
        cv_folds,
        seed=st.session_state.get("global_seed"),
    )

    stored_sig = st.session_state.get("ml_signature")
    ml_out = st.session_state.get("ml_output")
    models_md = st.session_state.get("ml_models_table_md")

    if ml_out is None or stored_sig is None:
        return "no_ml", "", None, {}

    if not models_md:
        # We have output but no table (e.g., intermediate or error); rare but helpful
        return "missing_table", "", None, {}

    if stored_sig != current_sig:
        diffs = _diff_signature(stored_sig, current_sig)
        reason = _human_reason(diffs)
        return "mismatch", "", None, {"reason": reason}

    # good to go
    return "ok", models_md, st.session_state.get("ml_rf_importances"), {}

def render_llm_tab(df: pd.DataFrame, default_name: str = "Dataset") -> None:
    """Streamlit UI for the LLM Report tab."""
    st.subheader("LLM Report Generation")

    dataset_name = st.text_input("Dataset name", value=default_name)

    include_modeling = st.checkbox("Include ML modeling in the report", value=False)

    #  Bind to ML tab target only
    target = st.session_state.get("ml_target")
    if target:
        st.caption(f"Target (from ML tab): **{target}**")
    else:
        st.info("No target selected yet. Pick a target and click **Run models** in the **ML Insights** tab first.")

    # --- Provider & model: read from unified sidebar (session_state), no UI here ---
    provider = st.session_state.get("llm_provider", "Ollama")
    ollama_model = st.session_state.get("ollama_model", "mistral")
    openai_model = st.session_state.get("openai_model", "gpt-4o-mini")
    openai_available = st.session_state.get("openai_available", False)

    active_model = openai_model if provider == "OpenAI" else ollama_model
    st.caption(f"Provider: **{provider}** · Model: **{active_model}**")

    # LLM factory with graceful fallback + quota guard
    def make_llm(provider_choice: str):
        if provider_choice == "OpenAI":
            if not openai_available:
                raise RuntimeError("OpenAI not configured. Install `langchain_openai` and set `OPENAI_API_KEY`.")
            return ChatOpenAI(model=openai_model, temperature=0.2)
        # Ollama path
        return OllamaLLM(model=ollama_model)
       
    # ---- PRE-CHECK & CACHE LOOKUP (no target picker here) ----
    models_table_md = ""
    feat_imps = None
    status, ctx = "ok", {}

    if include_modeling:
        if not target:
            status = "no_target"
            st.warning("No target from ML tab. Pick a target and click **Run models** there first.")
        else:
            status, models_table_md, feat_imps, ctx = cached_ml_artifacts(
                df,
                dataset_name or "Dataset",
                target
            )
            if status == "no_ml":
                st.info("No cached ML results yet. Please run models in **ML Insights** first.")
            elif status == "missing_table":
                st.info("Cached metrics table is missing. Please re-run models in **ML Insights**.")
            elif status == "mismatch":
                st.info(f"ML cache is out of date ({ctx.get('reason','settings changed')}). Please re-run models in **ML Insights**.")
            elif status == "ok":
                st.caption("Using results from **ML Insights** (last run).")
    else:
        st.caption("Tip: Leave modeling off to generate an EDA-only report.")

    # ---- BUTTON (enabled only when cache is valid) ----
    can_generate = (not include_modeling) or (status == "ok")
    clicked = st.button("Generate Report", disabled=not can_generate)

    if clicked:
        with st.spinner("Analyzing dataset with AI..."):
            report_text = llm_report_tab(
                df=df,
                dataset_name=dataset_name or "Dataset",
                target=target if include_modeling else None,
                feature_importances=feat_imps if include_modeling else None,
                models_table_md=models_table_md if include_modeling else "",
                excluded_columns=st.session_state.get("ml_excluded_cols", []),
                llm=make_llm(provider),
            )
            st.session_state["llm_report_text"] = report_text
    
    # ---- ALWAYS SHOW LAST REPORT ----
    if st.session_state.get("llm_report_text"):
        st.subheader("Generated Report")
        st.markdown(st.session_state["llm_report_text"])
        st.download_button(
            "Download report (Markdown)",
            data=st.session_state["llm_report_text"].encode("utf-8"),
            file_name=f"{dataset_name or 'dataset'}_report.md",
            mime="text/markdown",
        )
    else:
        st.info("No report yet. Configure options and click **Generate Report**.")

    