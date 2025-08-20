from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd

from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from langchain_core.output_parsers import StrOutputParser

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

# ---------- Main function: compute stats, call LLM ----------

def llm_report_tab(
    df: pd.DataFrame,
    dataset_name: str = "Dataset",
    target: Optional[str] = None,
    model_metrics: Optional[Dict[str, Dict[str, float]]] = None,
    feature_importances: Optional[Dict[str, float]] = None,
    recommendations: Optional[str] = None,
    llm_model_name: str = "mistral",
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
You are an AI data analyst. Write a structured, factual report based ONLY on the provided fields.
Do NOT guess or invent values. If a field is "not available", say so plainly.
Do NOT claim "data leakage" unless explicitly stated in the inputs.

Title: Data Analysis Report for {dataset_name}

## Dataset Overview
- Shape (rows, columns): {shape}
- Column types: 
{column_types}
- Example columns: {sample_columns}
- Class balance for target "{target}": 
{class_balance}

## Data Quality
- Missing values per column:
{missing_values}
- Potential issues: If missing values are high for any column, note the top offenders. Otherwise say "no major issues identified."

## Correlations & Feature Insights
- Key correlations with target (|corr| strongest first):
{correlations}
- Feature importances (if available):
{feature_importances}

## Model Performance
- Models compared: {model_list}
- Metrics (per model):
{model_metrics}

## Recommendations
{recommendations}
"""

    prompt = PromptTemplate(
        template=template.strip(),
        input_variables=[
            "dataset_name", "shape", "column_types", "sample_columns", "target",
            "class_balance", "missing_values", "correlations", "feature_importances",
            "model_list", "model_metrics", "recommendations"
        ],
    )

    # ---- Build inputs (as readable bullet lists / strings) ----
    inputs = {
        "dataset_name": dataset_name,
        "shape": str(shape),
        "column_types": _fmt_dict(col_types, max_items=50),
        "sample_columns": ", ".join(sample_columns) if sample_columns else "not available",
        "target": target or "not specified",
        "class_balance": _fmt_dict(class_balance, max_items=20) if class_balance else "not available",
        "missing_values": _fmt_dict(missing_vals, max_items=50) if missing_vals else "no missing values",
        "correlations": _fmt_dict(correlations, max_items=top_k_corr) if correlations else "not available",
        "feature_importances": feature_importances_str if feature_importances else "not available",
        "model_list": ", ".join(model_list) if model_list else "not available",
        "model_metrics": model_metrics_str if model_metrics else "not available",
        "recommendations": recommendations,
    }

    # ---- Runnable pipeline: prompt | llm | StrOutputParser ----
    llm = Ollama(model=llm_model_name)
    chain = prompt | llm | StrOutputParser()

    report: str = chain.invoke(inputs)
    return report
