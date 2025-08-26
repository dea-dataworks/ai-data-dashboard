import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

# --- Schema / Snapshot panel ---
def show_schema_panel(df: pd.DataFrame) -> None:
    """
    Read-only snapshot of the dataset: dtype counts, memory, overall missing, and
    top-5 high-cardinality text columns.
    """
    st.subheader("Dataset Snapshot")

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

    # High-cardinality text columns (top 5)
    if len(cat_cols) > 0:
        card = (
            df[cat_cols]
            .nunique(dropna=True)
            .sort_values(ascending=False)
            .head(5)
            .rename("unique_values")
        )
        st.markdown("**Top‑5 High‑Cardinality Text Columns**")
        st.dataframe(card.to_frame())
    else:
        st.info("No object/category/string columns detected.")
        
# summary
def show_summary(df):
    st.subheader("Basic Statistics")
    st.write("This table provides a statistical summary of your dataset, including count, mean, standard deviation, and quartile information for numerical columns, and unique counts and top values for categorical columns.")
    st.dataframe(df.describe(include="all").T)

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
    st.subheader("Correlation Heatmap")
    st.write("This heatmap visualizes the **correlation coefficients** between numerical features in your dataset. Values closer to **1** (red) indicate a strong positive correlation, values closer to **-1** (blue) indicate a strong negative correlation, and values near **0** (white/light colors) indicate a weak or no linear correlation.")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# distributions for numerical
def plot_distributions(df):
    st.subheader("Feature Distributions")
    st.write("This section allows you to visualize the distribution of numerical features using **histograms** and **Kernel Density Estimates (KDE)**. Histograms show the frequency of data points within specific bins, while KDE provides a smooth estimate of the data's probability density.")
    col = st.selectbox("Select a numeric column", df.select_dtypes(include=["int64", "float64"]).columns, key="eda_plot_distributions_numeric_select")
    fig, ax = plt.subplots()
    sns.histplot(df[col], bins=30, kde=True, ax=ax)
    st.pyplot(fig)
    plt.close(fig) 

def plot_categorical(df):
    st.subheader("Categorical Feature Distribution")
    st.write("This section visualizes the distribution of categorical features with **less than 20 unique values**. Each bar shows the count of occurrences for a specific category.")

    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    plottable_cat_cols = []

    for col in cat_cols:
        if df[col].nunique() < 20:
            plottable_cat_cols.append(col)

    if len(plottable_cat_cols) > 0:
        col = st.selectbox("Select a categorical column", plottable_cat_cols, key="eda_plot_categorical_select")
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No categorical columns detected.")

# --- Boxplot for numeric columns ---
def show_boxplot(df: pd.DataFrame) -> None:
    st.subheader("Outlier Detection (Boxplot)")
    num_cols = df.select_dtypes(include=["number"]).columns
    if len(num_cols) == 0:
        st.info("No numeric columns available for boxplot.")
        return
    col = st.selectbox("Select a numeric column for boxplot", num_cols, key="eda_show_boxplot_numeric_select")
    fig, ax = plt.subplots()
    sns.boxplot(x=df[col], ax=ax)
    st.pyplot(fig)
    plt.close(fig)

# --- Value counts for categorical columns ---
def show_value_counts(df: pd.DataFrame) -> None:
    st.subheader("Value Counts (Categorical)")
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    if len(cat_cols) == 0:
        st.info("No categorical columns available for value counts.")
        return
    col = st.selectbox("Select a categorical column", cat_cols, key="eda_show_value_counts_select")
    counts = df[col].value_counts(dropna=False).to_frame("count")
    st.dataframe(counts)

