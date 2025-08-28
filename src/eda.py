import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from matplotlib import font_manager as fm, rcParams
from pathlib import Path


# ---- Font ----
f = Path(__file__).resolve().parent / "fonts" / "Inter-VariableFont_opsz,wght.ttf"
FONT_STACK = ["DejaVu Sans"]
if f.exists():
    fm.fontManager.addfont(str(f))
    family = fm.FontProperties(fname=str(f)).get_name()  # typically "Inter"
    FONT_STACK = [family, "DejaVu Sans"]

mpl.rcParams.update({"font.family": FONT_STACK, "axes.titleweight": "semibold"})

# Alternative colors:
# palette: tableau
_TABLEAU10 = ["#4E79A7","#F28E2B","#E15759","#76B7B2","#59A14F",
              "#EDC948","#B07AA1","#FF9DA7","#9C755F","#BAB0AC"]

# palette: professional"
_PRO_PAL = ["#556EE6", "#23A699", "#E66E55", "#8B77AA", "#C3A634", "#3E4C59", "#9AA5B1"]

# grays and blues
grays_blues = ["#9AA5B1", "#4A90E2", "#2F5597", "#5B9BD5"]

# --- Global Dashboard Theme: Main accent color for all plots (hist bars, boxplots, bar charts, etc.) 
DASHBOARD_COLOR ="#5B9BD5"

def _is_dark_theme() -> bool:
    try:
        return (st.get_option("theme.base") or "").lower() == "dark"
    except Exception:
        return False

def get_style_params(compact: bool) -> dict:
    """
    Keep your original styling and visible compact shrink in places where figsize matters (e.g., EDA).
    Only change: unify histogram bins so compact vs normal doesn't *look* different due to bin count.
    Note: In two-column layouts (ML diagnostics), Streamlit width is fixed by the column,
    so figsize differences won't be noticeable — that's expected.
    """
    BINS = 30  # <-- unified bins (the only real change)

    return {
        "figsize": (6.5, 3.6) if compact else (8.8, 4.8),   
        "dpi": 120,
        "title_fs": 12 if compact else 14,
        "label_fs": 10 if compact else 12,
        "tick_fs": 9 if compact else 10,
        "lw": 1.1 if compact else 1.3,
        "bins": BINS,  # <-- unified
        "cmap": "Blues",
        "main_color": DASHBOARD_COLOR,
        "axes_face": "#22262e" if _is_dark_theme() else "#FAFAFA",
        "spine_color": "#9AA1AA" if _is_dark_theme() else "#B0B7BF",
        "bar_alpha": 0.90,
        "bar_edge_color": "#000000",
        "bar_edge_width": 0.7 if compact else 0.8,
    }


class PlotStyle:
    def __init__(self, compact: bool):
        self.p = get_style_params(compact)
        self._rc = None

    def __enter__(self):
        self._rc = mpl.rc_context({
            "figure.dpi": self.p["dpi"],
            "axes.titlesize": self.p["title_fs"],
            "axes.labelsize": self.p["label_fs"],
            "axes.facecolor": self.p["axes_face"],
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "xtick.labelsize": self.p["tick_fs"],
            "ytick.labelsize": self.p["tick_fs"],
            "font.family": FONT_STACK,
        })
        self._rc.__enter__()
        return self

    def __exit__(self, *exc):
        self._rc.__exit__(*exc)

def stylize_axes(ax, title=None, xlabel=None, ylabel=None, lw=1.3, grid_axis=None):
    # grid_axis: "x", "y", "both", or None
    if grid_axis:
        ax.grid(axis=grid_axis, linestyle="--", alpha=0.22)
    else:
        ax.grid(False)

    # spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(0.8)

    if title:  ax.set_title(title, pad=8)
    if xlabel: ax.set_xlabel(xlabel, labelpad=6)
    if ylabel: ax.set_ylabel(ylabel, labelpad=6)

    for line in ax.lines:
        line.set_linewidth(lw)


# --- Schema / Snapshot panel ---
def show_schema_panel(df: pd.DataFrame) -> None:
    """
    Read-only snapshot of the dataset: dtype counts, memory, overall missing.
    (top-5 high-cardinality text columns moved to Summary)
    """
    #st.subheader("Dataset Snapshot")

    # Type counts
    num_cols = df.select_dtypes(include=["number"]).columns
    cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
    bool_cols = df.select_dtypes(include=["bool"]).columns
    dt_cols  = df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Rows", f"{df.shape[0]:,}")
    with c2:
        st.metric("Columns", f"{df.shape[1]:,}")
    with c3:
        st.metric("Numeric", f"{len(num_cols)}")
    with c4:
        st.metric("Categorical", f"{len(cat_cols)}")
    with c5:
        st.metric("Datetime/Bool", f"{len(dt_cols)}/{len(bool_cols)}")

    # Memory + overall missing
    mem_mb = df.memory_usage(deep=True).sum() / (1024 ** 2)
    overall_missing_pct = (df.isna().sum().sum() / df.size * 100) if df.size else 0.0

    c6, c7 = st.columns(2)
    with c6:
        st.metric("Memory (MB)", f"{mem_mb:.2f}")
    with c7:
        st.metric("Overall Missing (%)", f"{overall_missing_pct:.2f}%")

            
# summary
def show_summary(df: pd.DataFrame):
    # st.subheader("Basic Statistics")
    st.write(
        "Quick numerical/categorical summary below. On the right, see high-cardinality text columns; "
        "on the left, columns with missing values."
    )
    # Main stats table
    st.dataframe(
        df.describe(include="all").T,
        use_container_width=True,
        height=320
    )

    # --- Side-by-side compact tables: Missing | High-Cardinality ---
    c1, c2 = st.columns(2)

    # Missing table (Count + %)
    with c1:
        st.markdown("**Missing values by column**")
        missing_counts = df.isnull().sum()
        missing_df = missing_counts[missing_counts > 0].sort_values(ascending=False).to_frame("Count")
        if not df.empty:
            total_rows = len(df)
        else:
            total_rows = 0
        if total_rows > 0 and not missing_df.empty:
            missing_df["Pct %"] = (missing_df["Count"] / total_rows * 100).round(2)
            st.dataframe(missing_df, use_container_width=True, height=260)
        else:
            st.info("No missing values found in the dataset! 🎉")

    # High-cardinality table (Top-5 text columns)
    with c2:
        st.markdown("**Top-5 High-Cardinality Text Columns**")
        cat_cols = df.select_dtypes(include=["object", "category", "string"]).columns
        if len(cat_cols) > 0:
            card = (
                df[cat_cols]
                .nunique(dropna=True)
                .sort_values(ascending=False)
                .head(5)
                .rename("unique_values")
                .to_frame()
            )
            st.dataframe(card, use_container_width=True, height=260)
        else:
            st.info("No object/category/string columns detected.")


# missing value
def show_missing(df):
    st.subheader("Missing Values")
    st.write("This table identifies columns with **missing data** (NaN values) and shows the **count of missing entries** for each. Columns not listed here have no missing values.")
    
    missing_counts = df.isnull().sum()

    # Filter for columns with missing values and convert to DataFrame
    missing_df = missing_counts[missing_counts > 0].to_frame()

    # Rename the column from '0' to 'Missing Count'
    missing_df.columns = ['Count']

    # Display only if there are missing values
    if not missing_df.empty:
        st.dataframe(missing_df)
    else:
        st.info("No missing values found in the dataset! 🎉")

# correlation
def show_correlation(df):
    # st.subheader("Correlation Heatmap")
    st.write(
        "This heatmap visualizes the **correlation coefficients** between numerical features. "
        "Values close to **1** indicate strong positive correlation; **-1** strong negative."
    )

    corr = df.corr(numeric_only=True)
    if corr.empty:
        st.info("No numerical columns to correlate.")
        return

    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        sns.heatmap(
            corr,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap=params["cmap"],        # unified palette
            cbar=True,
            linewidths=0.3,
            linecolor="#00000010",
            square=False
        )
        stylize_axes(
            ax,
            title="Correlation Heatmap",
            xlabel=None,
            ylabel=None,
            lw=params["lw"],
            grid_axis=None
        )
        # Make tick labels a touch smaller/tilted for compactness
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0)

        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# distributions for numerical
def plot_distributions(df):
    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    st.write(
        "This section allows you to visualize the distribution of numerical "
        "features using **histograms** and **Kernel Density Estimates (KDE)**. "
        "Histograms show the frequency of data points within specific bins, "
        "while KDE provides a smooth estimate of the data's probability density."
    )

    col = st.selectbox(
        "Select a numeric column",
        df.select_dtypes(include=["int64", "float64"]).columns,
        key="eda_plot_distributions_numeric_select",
    )

    fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
    with PlotStyle(compact):
        ax.hist(
            df[col].dropna(),
            bins=params["bins"],                 # now unified via get_style_params
            color=params["main_color"],                  
            edgecolor=params["bar_edge_color"],          
            linewidth=params["bar_edge_width"],
            alpha=params["bar_alpha"]
        )
        stylize_axes(
            ax,
            title=f"Distribution — {col}",
            xlabel=col,
            ylabel="Count",
            lw=params["lw"],
            grid_axis="y"
        )

    st.pyplot(fig, use_container_width=False)
    plt.close(fig)


def plot_categorical(df):
    # st.subheader("Categorical Feature Distribution")
    st.write("Counts for categorical features with **< 20 unique values**.")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    plottable_cat_cols = [c for c in cat_cols if df[c].nunique(dropna=False) < 20]

    if not plottable_cat_cols:
        st.info("No categorical columns detected.")
        return

    col = st.selectbox("Select a categorical column", plottable_cat_cols, key="eda_plot_categorical_select")

    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    counts = df[col].value_counts(dropna=False)

    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        ax.bar(
            counts.index.astype(str),
            counts.values,
            color=params["main_color"],
            edgecolor=params["bar_edge_color"],
            linewidth=params["bar_edge_width"],
            alpha=params["bar_alpha"]
        )
        stylize_axes(
            ax,
            title=f"Categorical Distribution — {col}",
            xlabel=col,
            ylabel="Count",
            lw=params["lw"],
            grid_axis="y"
        )
        ax.tick_params(axis="x", rotation=30, labelrotation=30)
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

# --- Boxplot for numeric columns ---
def show_boxplot(df: pd.DataFrame) -> None:
    # Numeric (exclude bool)
    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    num_cols = [c for c in num_cols if df[c].dtype != bool]

    # Exclude binary numeric (<= 2 unique)
    cols_for_box = [c for c in num_cols if df[c].dropna().nunique() > 2]

    if not cols_for_box:
        st.info("No non-binary numeric columns available for boxplot.")
        return

    col = st.selectbox(
        "Select a numeric column for boxplot",
        cols_for_box,
        key="eda_show_boxplot_numeric_select"
    )

    data = df[col].dropna()
    is_binary = data.nunique() <= 2  # will be False now; keep for safety

    compact = st.session_state.get("compact_mode", False)
    params = get_style_params(compact)

    with PlotStyle(compact):
        fig, ax = plt.subplots(figsize=params["figsize"], dpi=params["dpi"])
        sns.boxplot(
            x=data,
            ax=ax,
            color=params["main_color"],
            linewidth=params["bar_edge_width"],
            showfliers=not is_binary,
            fliersize=0 if is_binary else (2.5 if compact else 3.5),
        )
        stylize_axes(
            ax,
            title=f"Boxplot — {col}",
            xlabel=col,
            ylabel=None,
            lw=params["lw"],
            grid_axis="y",
        )
        st.pyplot(fig, use_container_width=not compact)
        plt.close(fig)

# --- Value counts for categorical columns ---
def show_value_counts(df: pd.DataFrame) -> None:
    # st.subheader("Value Counts (Categorical)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        st.info("No categorical columns available for value counts.")
        return
    col = st.selectbox("Select a categorical column", cat_cols, key="eda_show_value_counts_select")
    counts = df[col].value_counts(dropna=False).to_frame("count")
    st.dataframe(counts)


# --- Data quality warnings: Duplicates count, leakage checks (feature == target,|corr|≥0.95 numeric) ---
def show_data_quality_warnings(df: pd.DataFrame, target: str | None = None) -> None:
    #st.subheader("Data Quality Warnings")

    issues = []

    # Duplicate rows
    dup_count = df.duplicated().sum()
    if dup_count > 0:
        issues.append(f"⚠️ Found {dup_count} duplicate rows.")

    # Target leakage: identical columns
    if target and target in df.columns:
        for col in df.columns:
            if col != target and df[col].equals(df[target]):
                issues.append(f"⚠️ Column **{col}** is identical to target **{target}**.")

    # High correlation with target (numeric only)
    if target and target in df.columns:
        if pd.api.types.is_numeric_dtype(df[target]):
            corr = df.corr(numeric_only=True)
            if target in corr.columns:
                high_corr = corr[target].drop(target).abs()
                suspicious = high_corr[high_corr >= 0.95]
                for col, val in suspicious.items():
                    issues.append(f"⚠️ Column **{col}** is highly correlated with target (r={val:.2f}).")

    if issues:
        for msg in issues:
            st.warning(msg)
    else:
        st.success("No major data quality issues detected.")
