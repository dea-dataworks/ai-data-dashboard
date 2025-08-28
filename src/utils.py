# src/utils.py
from __future__ import annotations
import os
import pandas as pd
from datetime import datetime
from io import StringIO, BytesIO
import streamlit as st

from .data_preprocess import preprocess_df
from .ml_models import train_and_evaluate
from .eda import get_style_params, PlotStyle, stylize_axes
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay

# ---------- EDA text utilities ----------

def generate_summary_stats(df: pd.DataFrame, max_cols: int = 15) -> str:
    """
    Compact textual summary for LLM context.
    - Basic shape
    - Column dtypes
    - Head of describe() truncated
    """
    lines = []
    lines.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    dtypes = df.dtypes.astype(str)
    # Show up to max_cols dtype entries to avoid huge walls of text
    dtypes_view = dtypes.head(max_cols).to_string()
    if df.shape[1] > max_cols:
        dtypes_view += f"\n... ({df.shape[1]-max_cols} more columns)"
    lines.append("Column dtypes:\n" + dtypes_view)

    # Describe with include='all' can be big; cap columns for readability
    desc = df.describe(include="all").T
    if len(desc) > max_cols:
        desc = desc.head(max_cols)
        desc_text = desc.to_string()
        desc_text += f"\n... ({df.shape[1]-max_cols} more columns not shown)"
    else:
        desc_text = desc.to_string()
    lines.append("Describe (truncated):\n" + desc_text)

    return "\n".join(lines)


def generate_missing_values(df: pd.DataFrame, top_k: int = 20) -> str:
    """
    Return top columns by missing counts as text for LLM context.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if missing.empty:
        return "No missing values."
    if len(missing) > top_k:
        missing = missing.head(top_k)
        tail = " (truncated)"
    else:
        tail = ""
    return "Top missing-value columns:\n" + missing.to_string() + tail


def generate_correlation(df: pd.DataFrame, top_k: int = 15) -> str:
    """
    Return strongest absolute pairwise correlations among numeric columns (excluding self-corr).
    """
    num = df.select_dtypes(include=["number"])
    if num.shape[1] < 2:
        return "Not enough numeric columns for correlations."

    corr = num.corr(numeric_only=True)
    # Flatten upper triangle into a series of pairs
    pairs = []
    cols = corr.columns
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append(((cols[i], cols[j]), abs(corr.iloc[i, j]), corr.iloc[i, j]))
    if not pairs:
        return "No correlations found."

    # Sort by absolute correlation, descending
    pairs.sort(key=lambda x: x[1], reverse=True)
    top = pairs[:top_k]

    buf = StringIO()
    buf.write("Top correlations (abs value desc):\n")
    for (a, b), abs_val, signed in top:
        buf.write(f"- {a} ↔ {b}: r = {signed:.3f} (|r|={abs_val:.3f})\n")
    if len(pairs) > top_k:
        buf.write(f"... ({len(pairs)-top_k} more pairs)\n")
    return buf.getvalue().strip()


# ---------- Modeling summary utility ----------

def run_models(df: pd.DataFrame, target: str) -> dict:
    """
    Wrapper that:
      - preprocesses (using your existing preprocess_df)
      - trains/evaluates (using your existing train_and_evaluate)
      - returns a dict with:
          'task_type', 'results', and 'text_summary' for LLM context.
    """
    X, y = preprocess_df(df, target)
    out = train_and_evaluate(X, y, target)
    task_type = out["task_type"]
    results = out["results"]

    # Convert results into a concise, human-readable text
    lines = [f"Detected task: {task_type}"]
    if task_type == "classification":
        for model, m in results.items():
            acc = m.get("accuracy")
            f1  = m.get("f1_score")
            auc = m.get("roc_auc")
            lines.append(
                f"- {model}: Accuracy={acc:.3f} | F1 (weighted)={f1:.3f} | ROC AUC={(f'{auc:.3f}' if auc is not None else 'nan')}"
            )
    else:
        for model, m in results.items():
            r2  = m.get("r2_score")
            mae = m.get("mae")
            rmse = m.get("rmse")
            lines.append(
                f"- {model}: R²={r2:.3f} | MAE={mae:.3f} | RMSE={rmse:.3f}"
            )

    return {
        "task_type": task_type,
        "results": results,
        "text_summary": "\n".join(lines)
    }

def plot_confusion_matrix(y_true, y_pred, labels):
    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=labels, cmap=params["cmap"], ax=ax
        )
        stylize_axes(ax, title="Confusion Matrix", xlabel="Predicted", ylabel="Actual", lw=params["lw"], grid_axis=None)
    return fig

def plot_roc_curve(y_true, y_proba):
    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
        # Style tweaks: thicker lines, consistent color for the model curve
        for line in ax.get_lines():
            line.set_linewidth(params["lw"] + 0.5)
        # Diagonal baseline stays dashed grey; model curve adopts main color
        if len(ax.get_lines()) >= 1:
            ax.get_lines()[0].set_color(params["main_color"])
        stylize_axes(ax, title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", lw=params["lw"], grid_axis=None)
    return fig

def plot_regression_diagnostics(y_true, y_pred):
    figs = []
    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    # Residuals vs Fitted
    with PlotStyle(compact):
        fig1, ax1 = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        residuals = y_true - y_pred
        ax1.scatter(y_pred, residuals, alpha=0.6, color=params["main_color"], edgecolors="none")
        ax1.axhline(0, color="#666666", linestyle="--", linewidth=params["lw"])
        stylize_axes(ax1, title="Residuals vs Fitted", xlabel="Predicted", ylabel="Residuals", lw=params["lw"], grid_axis="y")
        figs.append(fig1)

    # Prediction Error Plot
    with PlotStyle(compact):
        fig2, ax2 = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        ax2.scatter(y_true, y_pred, alpha=0.6, color=params["main_color"], edgecolors="none")
        lo, hi = float(min(y_true.min(), y_pred.min())), float(max(y_true.max(), y_pred.max()))
        ax2.plot([lo, hi], [lo, hi], "--", color="#666666", linewidth=params["lw"])
        stylize_axes(ax2, title="Prediction Error Plot", xlabel="Actual", ylabel="Predicted", lw=params["lw"], grid_axis=None)
        figs.append(fig2)

    return figs

def plot_feature_importances(importances: dict, top_n: int = 10):
    if not importances:
        return None

    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    s = pd.Series(importances).sort_values(ascending=False).head(top_n)
    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        ax.barh(
            s.index.astype(str),
            s.values,
            color=params["main_color"],
            edgecolor=params.get("bar_edge_color", "#00000020"),
            linewidth=params.get("bar_edge_width", 0.8),
            alpha=params.get("bar_alpha", 0.95),
        )
        ax.invert_yaxis()
        stylize_axes(ax, title=f"Top {top_n} Features (Random Forest)", xlabel="Importance", ylabel=None, lw=params["lw"], grid_axis="x")
    return fig

# ---------- Exporting utility ----------
def _ts():
    return datetime.now().strftime("%Y%m%d_%H%M")

def _slugify(s: str) -> str:
    return "".join(c.lower() if c.isalnum() else "-" for c in s).strip("-")

def df_download_buttons(title: str, df: pd.DataFrame, base: str = "dashboard", excel: str = "off"):
    """
    Show CSV + (optionally) Excel download buttons for a DataFrame.
    excel: "on" | "off"
    """
    if df is None or df.empty:
        return
    name = f"{_slugify(base)}-{_slugify(title)}-{_ts()}"

    # CSV (primary)
    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        f"⬇️ Download {title} (CSV)",
        data=csv_bytes,
        file_name=f"{name}.csv",
        mime="text/csv",
        use_container_width=True
    )

    # Excel (secondary, only if enabled)
    if excel == "on":
        xbuf = BytesIO()
        with pd.ExcelWriter(xbuf, engine="openpyxl") as writer:
            sheet = _slugify(title)[:28] or "data"
            df.to_excel(writer, index=False, sheet_name=sheet)
        xbuf.seek(0)
        st.download_button(
            f"⬇️ Download {title} (Excel)",
            data=xbuf,
            file_name=f"{name}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )

def fig_download_button(title: str, fig, base: str = "dashboard", dpi: int = 150):
    """Show a PNG download button for a Matplotlib figure."""
    if fig is None:
        return
    name = f"{_slugify(base)}-{_slugify(title)}-{_ts()}.png"
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    st.download_button(
        f"🖼️ Download {title} (PNG)",
        data=buf,
        file_name=name,
        mime="image/png",
        use_container_width=True
    )

# --- Sidebar Helpers ---
def _openai_is_available() -> bool:
    try:
        from langchain_openai import ChatOpenAI  # noqa: F401
        pkg_ok = True
    except Exception:
        pkg_ok = False
    key = None
    try:
        key = st.secrets["OPENAI_API_KEY"]  # type: ignore[index]
    except Exception:
        key = os.getenv("OPENAI_API_KEY")
    return bool(pkg_ok and key)

def sidebar_global_settings():
    st.header("⚙️ Global Settings")
    st.session_state["compact_mode"] = st.checkbox("Compact mode (smaller plots, tighter layout)", value=False)
    st.session_state["collapse_plots"] = st.checkbox("Collapse plots by default", value=False)
    st.session_state["global_seed"] = st.number_input("Random seed", min_value=0, value=42, step=1)
    st.session_state["cv_folds"] = st.number_input("CV folds", min_value=2, max_value=10, value=5, step=1)
    st.sidebar.checkbox("Show Excel downloads",value=st.session_state.get("dl_excel_global", False), key="dl_excel_global",
    help="When on, all tables show an Excel button alongside CSV.")

    

def sidebar_llm_settings():
    st.subheader("🤖 LLM Provider")
    openai_available = _openai_is_available()
    help_txt = None if openai_available else (
        "OpenAI disabled (missing package or API key). Install `langchain_openai` and set OPENAI_API_KEY."
    )
    provider = st.radio("Provider", ["Ollama", "OpenAI"], index=0, help=help_txt)
    if provider == "OpenAI" and not openai_available:
        st.warning("OpenAI is not configured; falling back to **Ollama**.")
        provider = "Ollama"

    if provider == "Ollama":
        ollama_model = st.selectbox("Ollama model", options=["mistral"], index=0)
        openai_model = "gpt-4o-mini"
    else:
        openai_model = st.selectbox("OpenAI model", options=["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-4.1"], index=0)
        ollama_model = "mistral"

    # expose globally
    st.session_state["llm_provider"] = provider
    st.session_state["ollama_model"] = ollama_model
    st.session_state["openai_model"] = openai_model
    st.session_state["openai_available"] = openai_available


# ---- Report Helpers ---
def ml_signature(df, dataset_name, target, excluded_cols, cv_used, cv_folds):
    return (
        dataset_name,
        df.shape,
        tuple(sorted(df.columns)),
        target,
        tuple(sorted(excluded_cols or [])),
        bool(cv_used),
        int(cv_folds) if cv_used else None,
    )
