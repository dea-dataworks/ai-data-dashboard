import streamlit as st
import pandas as pd

import src.eda as eda
import src.ml_models as ml
from src import data_preprocess

from src.data_preprocess import preprocess_df
from src.ml_models import train_and_evaluate

# Set page configuration
st.set_page_config(page_title="AI Data Insight Dashboard", layout="centered")
st.title("AI Data Insight Dashboard")

# Function to reset the application 
def reset_app():
    st.session_state.clear() 
    st.session_state.df = None

# --- Data Loading Section ---
# Initialize 'df' in session_state if it doesn't exits
if "df" not in st.session_state:
    st.session_state.df = None

# Display file uploader if no DataFrame is loaded yet
if st.session_state.df is None:
    uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

    if uploaded_file:
        try:          
            if uploaded_file.name.endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.session_state.df = df    # Store the dataframe in session state
            st.success("File uploaded and processed sucessfully!")
            st.rerun()                  # Rerun to display the content immediately with the new df
        except Exception as e:
            st.error(f"Error reading file: {e}. Please ensure it's a valid CSV or Excel file.")
            st.session_state.df = None  # Clear df on error
            st.stop()                   # Stop execution if file reading fails

else:
    st.info("Dataset loaded. Use the tabs below to explore your data.")

# --- Application Tabs Section ---
# Only display tabs if a DataFrame is available in session state
if st.session_state.df is not None:
    df = st.session_state.df 

    tab1, tab2, tab3, tab4 = st.tabs(["Upload & Preview", "EDA", "ML Insights", "LLM Report"])

    with tab1:
        st.subheader("Dataset Preview")
        st.write(f"Shape: {df.shape[0]} rows x {df.shape[1]} columns")
        st.dataframe(df.head())

    with tab2:
        st.subheader("Exploratory Data Analysis")
        eda.show_summary(df)
        st.markdown("---") # Visual separator
        eda.show_missing(df)
        st.markdown("---")
        eda.show_correlation(df)
        st.markdown("---")
        eda.plot_distributions(df)
        st.markdown("---")
        eda.plot_categorical(df)

    with tab3:
        st.subheader("Machine Learning")

        if df is not None:
            target = st.selectbox("Select target variable", df.columns)
            if target:
                try:
                    # Preprocess: safe step (drops high-cardinality, splits X/y)
                    X, y = data_preprocess.preprocess_df(df, target)

                    # Train models + evaluate
                    output = ml.train_and_evaluate(X, y, target)

                    st.write(f"### Detected Task: {output['task_type'].capitalize()}")

                    # Display results per model
                    for model, metrics in output["results"].items():
                        st.subheader(model)
                        for k, v in metrics.items():
                            if isinstance(v, float):
                                st.write(f"- **{k.replace('_',' ').title()}**: {v:.3f}")
                            elif isinstance(v, dict):
                                # Convert classification report dict → DataFrame for clean table
                                report_df = pd.DataFrame(v).transpose()
                                report_df = report_df.round(3)  # round numbers
                                st.dataframe(report_df)
                            elif v is not None:
                                st.write(f"- **{k.replace('_',' ').title()}**: {v}")

                except ValueError as e:
                    st.error(str(e))
                except Exception as e:
                    st.error(f"Unexpected error: {e}")

           
    with tab4:
        st.subheader("LLM Report Generation")
        st.info("Leverage large language models to generate reports on your data here!")

# --- Reset Button Section ---
# Add reset button at the bottom, only if a file has been processed
if st.session_state.df is not None:
    st.markdown("---") # Add a separator before the reset button for better UI
    if st.button("Reset Application"):
        reset_app()
        st.rerun() # Rerun the app to reflect the reset state

#streamlit run ai-data-dashboard/app.py