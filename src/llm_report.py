import streamlit as st
import pandas as pd
from langchain.llms import Ollama
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Import from your existing modules
from src.data_preprocess import preprocess_df
from src.eda import generate_summary_stats, generate_missing_values, generate_correlation
from src.ml_models import run_models

def generate_report(df: pd.DataFrame):
    """Generate a text-based analysis of the dataset using LLM."""
    
    # Step 1: EDA on raw
    summary = generate_summary_stats(df)
    missing = generate_missing_values(df)
    corr = generate_correlation(df)


    # Step 2: Preprocess only for modeling
    df_clean = preprocess_df(df)
    models_output = run_models(df_clean)

    # Step 3: Turn into text for the LLM
    context = f"""
    DATA SUMMARY:
    {summary}

    MISSING VALUES:
    {missing}

    CORRELATIONS:
    {corr}

    MODEL RESULTS:
    {models_output}
    """

    # Step 4: LLM to write report
    template = """
    You are a data science assistant. Based on the dataset information below,
    write a structured report with sections:
    1. Dataset Overview
    2. Data Quality (missing values, correlations)
    3. Model Insights (classification/regression results)
    4. Recommendations

    Dataset info:
    {context}
    """

    prompt = PromptTemplate(template=template, input_variables=["context"])
    llm = Ollama(model="mistral")  # or whatever you’re using
    chain = LLMChain(llm=llm, prompt=prompt)

    return chain.run(context=context)


def llm_report_tab(df: pd.DataFrame):
    """Streamlit tab for LLM report generation."""
    st.header("📊 AI-Generated Data Report")

    if df is None:
        st.info("Upload and process a dataset first.")
        return

    if st.button("Generate Report"):
        with st.spinner("Analyzing dataset with AI..."):
            report = generate_report(df)
            st.subheader("Generated Report")
            st.write(report)
