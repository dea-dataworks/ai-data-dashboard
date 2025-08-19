import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

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
    col = st.selectbox("Select a numeric column", df.select_dtypes(include=["int64", "float64"]).columns)
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
        col = st.selectbox("Select a categorical column", plottable_cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        ax.set_ylabel("Count")
        st.pyplot(fig)
    else:
        st.info("No categorical columns detected.")


