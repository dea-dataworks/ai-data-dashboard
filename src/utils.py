# src/utils.py
from __future__ import annotations
import pandas as pd
from io import StringIO

from .data_preprocess import preprocess_df
from .ml_models import train_and_evaluate
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
    fig, ax = plt.subplots()
    #disp = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax)
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=labels, cmap="Blues", ax=ax)
    #plt.close(fig)

    return fig

def plot_roc_curve(y_true, y_proba):
    fig, ax = plt.subplots()
    RocCurveDisplay.from_predictions(y_true, y_proba, ax=ax)
    #plt.close(fig)
    return fig

def plot_regression_diagnostics(y_true, y_pred):
    figs = []

    # Residuals vs fitted
    fig1, ax1 = plt.subplots()
    residuals = y_true - y_pred
    ax1.scatter(y_pred, residuals, alpha=0.6)
    ax1.axhline(0, color="red", linestyle="--")
    ax1.set_xlabel("Predicted")
    ax1.set_ylabel("Residuals")
    ax1.set_title("Residuals vs Fitted")
    figs.append(fig1)

    # y vs yhat
    fig2, ax2 = plt.subplots()
    ax2.scatter(y_true, y_pred, alpha=0.6)
    ax2.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--")
    ax2.set_xlabel("Actual")
    ax2.set_ylabel("Predicted")
    ax2.set_title("Prediction Error Plot")
    figs.append(fig2)

    return figs