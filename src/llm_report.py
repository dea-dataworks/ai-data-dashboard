from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser

import streamlit as st
from typing import Optional
from src.utils import run_models

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
    correlations = _safe_corr_with_target(df, target, top_k=top_k_corr) if target else {}
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

    # Default recs if none provided
    if not recommendations:
        recs = []
        if missing_vals:
            worst = list(missing_vals.items())[0]
            recs.append(f"Handle missing values (e.g., impute or drop). Highest missing: {worst[0]}={worst[1]}.")
        if target:
            recs.append(f"Check class balance for '{target}' and consider stratified splits/metrics beyond accuracy.")
        if correlations:
            top_feat, top_val = next(iter(correlations.items()))
            recs.append(f"Leverage strongly correlated features (e.g., {top_feat} with corr {top_val:+.3f}).")
        recs.append("Consider feature engineering and model calibration; validate with cross-validation.")
        recommendations = "- " + "\n- ".join(recs)

    # Model lists / metrics formatting
    model_list: List[str] = list(model_metrics.keys()) if model_metrics else []
    model_metrics_str = _fmt_dict(model_metrics or {}, max_items=20)
    feature_importances = _top_items(feature_importances or {}, k=15)
    feature_importances_str = _fmt_dict(feature_importances, max_items=15)

    # ---- Prompt template (tight, structured, no guessing) ----
    template = """
            You are a precise data analyst. Produce a concise, client-ready Markdown report.
            Follow these rules strictly:
            - Express shape as "rows × columns" (not a Python tuple).
            - Group variables by meaning when possible; do not dump long lists.
            - Write correlations in plain English; include r only in parentheses.
            - Avoid hedging like "if significant"; be direct and actionable.
            - Fix typos. Use short sentences and bullets.

            # Data Analysis Report for {dataset_name}

            ## Dataset Overview
            - Shape: {shape}
            - Key variables (brief): {sample_columns}
            - Class balance for target "{target}":
            {class_balance}

            ## Data Quality
            - Missing values (top offenders):
            {missing_values}

            ## Correlations & Feature Insights
            {correlations}
            - Feature importances (if available):
            {feature_importances}

            ## Model Performance
            - One-sentence comparison: which model is best and by roughly how much (accuracy/F1/AUC).
            - Metrics (per model):

            {models_table_md}

            ## Recommendations
            Provide 4–6 concrete steps tailored to this dataset. Be specific (what to impute, what to drop, what to engineer, how to validate).
            """

    prompt = PromptTemplate(
        template=template.strip(),
        input_variables=[
            "dataset_name", "shape", "column_types", "sample_columns", "target", 
            "class_balance", "missing_values", "correlations", "feature_importances",
            "model_list", "model_metrics", "recommendations", "models_table_md"
        ],
    )

    # ---- Build inputs (as readable bullet lists / strings) ----
    inputs = {
        "dataset_name": dataset_name,
        "models_table_md": models_table_md or "Not run.",
        #"shape": str(shape),
        "shape": _shape_pretty(df),
        "column_types": _fmt_dict(col_types, max_items=50),
        "sample_columns": _key_vars_human(df),
        #"sample_columns": ", ".join(sample_columns[:7]) if sample_columns else "not available", 
        "target": target or "not specified",
        "class_balance": _fmt_dict(class_balance, max_items=20) if class_balance else "not available",
        "missing_values": _fmt_dict(missing_vals, max_items=50) if missing_vals else "no missing values",
        "correlations": correlations_sent,
        "feature_importances": feature_importances_str if feature_importances else "not available",
        "model_list": ", ".join(model_list) if model_list else "not available",
        "model_metrics": model_metrics_str if model_metrics else "not available",
        "recommendations": recommendations,
    }

    # ---- Runnable pipeline: prompt | llm | StrOutputParser ----
    llm = OllamaLLM(model=llm_model_name)
    chain = prompt | llm | StrOutputParser()

    report: str = chain.invoke(inputs)
    return report

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


def render_llm_tab(df: pd.DataFrame, default_name: str = "Dataset") -> None:
    """Streamlit UI for the LLM Report tab."""
    st.subheader("LLM Report Generation")

    # Dataset name (prefill from uploaded filename if available)
    dataset_name = st.text_input("Dataset name", value=default_name)

    # Optional modeling
    include_modeling = st.checkbox("Include ML modeling in the report", value=False)

    target: Optional[str] = None
    model_metrics = None
    feature_importances = None  # (optional later)

    if include_modeling:
        # Let user choose the target
        target = st.selectbox("Select target column", options=list(df.columns))
        st.caption("Tip: choose your label column for classification/regression.")

    models_table_md = ""
    if st.button("Generate Report"):
        with st.spinner("Analyzing dataset with AI..."):
            # If modeling requested and target chosen, run models to get metrics text
            if include_modeling and target:
                out = run_models(df, target)  # returns task_type, results, text_summary
                models_table_md = _format_model_metrics(out)
                # Convert results into a compact metrics dict per model for the prompt
                # Classification vs regression keys differ; handle both
                metrics_dict = {}
                if out["task_type"] == "classification":
                    for model, m in out["results"].items():
                        metrics_dict[model] = {
                            "accuracy": round(m.get("accuracy", float("nan")), 3),
                            "f1_weighted": round(m.get("f1_score", float("nan")), 3),
                            "roc_auc": (round(m["roc_auc"], 3) if m.get("roc_auc") is not None else "na"),
                        }
                else:
                    for model, m in out["results"].items():
                        metrics_dict[model] = {
                            "r2": round(m.get("r2_score", float("nan")), 3),
                            "mae": round(m.get("mae", float("nan")), 3),
                            "rmse": round(m.get("rmse", float("nan")), 3),
                        }
                model_metrics = metrics_dict

            # Build the narrative using the existing function
            report_text = llm_report_tab(
                df=df,
                dataset_name=dataset_name or "Dataset",
                target=target,
                model_metrics=model_metrics,
                feature_importances=feature_importances,
                llm_model_name="mistral",
                top_k_corr=10,
                # NEW
                models_table_md=models_table_md,
            )

            st.subheader("Generated Report")
            st.markdown(report_text)

            st.download_button(
                label="Download report (Markdown)",
                data=report_text.encode("utf-8"),
                file_name=f"{dataset_name or 'dataset'}_report.md",
                mime="text/markdown"
            )