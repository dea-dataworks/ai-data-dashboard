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
                    # Preprocess (safe step: drops high-cardinality, splits X/y)
                    X, y = data_preprocess.preprocess_df(df, target)

                    # Train models + evaluate
                    output = ml.train_and_evaluate(X, y, target)
                    task_type = output["task_type"]

                    st.write(f"### Detected Task: {task_type.capitalize()}")

                    # ---- Classification ----
                    if task_type == "classification":
                        summary_data = []
                        for model, metrics in output["results"].items():
                            if "classification_report" in metrics and isinstance(metrics["classification_report"], dict):
                                f1_weighted = metrics["classification_report"]["weighted avg"]["f1-score"]
                            else:
                                f1_weighted = metrics.get("f1_score", None)

                            summary_data.append({
                                "Model": model,
                                "Accuracy": metrics.get("accuracy", None),
                                "F1 Score (weighted)": f1_weighted,
                                "ROC AUC": metrics.get("roc_auc", None),
                            })

                        # I will save this summary_df in case I need it later
                        summary_df = pd.DataFrame(summary_data)

                        # round and convert to string to force formatting
                        summary_df_table = summary_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

                        st.write("#### Performance Summary")
                        st.table(summary_df_table.set_index("Model"))
                        
                        # Advanced Metrics
                        with st.expander("🔍 Advanced Metrics (per-class details)"):
                            st.markdown("Note: `0`, `1`, etc. in **Class/Avg** are your target classes; rows like **macro avg**/**weighted avg** are aggregates.")
                            for model, metrics in output["results"].items():
                                st.markdown(f"**{model}**")
                                if "classification_report" in metrics and isinstance(metrics["classification_report"], dict):
                                    report_df = pd.DataFrame(metrics["classification_report"]).transpose()
                                    report_df = report_df.round(3)
                                    st.dataframe(report_df)
                                else:
                                    st.write("No detailed report available.")

                        # Metric Definitions
                        with st.expander("📖 Metric Definitions"):
                            st.markdown("""
                            - **Accuracy**: Proportion of correctly classified samples.
                            - **F1 Score (weighted)**: Harmonic mean of precision and recall, weighted by class frequency.
                            - **ROC AUC**: Measures how well the model separates classes; 0.5 = random, 1.0 = perfect.
                            - **Precision**: Among predicted positives, proportion that were actually positive.
                            - **Recall (Sensitivity)**: Among actual positives, proportion predicted correctly.
                            - **Macro Avg**: Average across classes, treating each equally.
                            - **Weighted Avg**: Average across classes, weighted by number of samples.
                            """)

                    # ---- Regression ----
                    elif task_type == "regression":
                        summary_data = []
                        for model, metrics in output["results"].items():
                            summary_data.append({
                                "Model": model,
                                "R²": metrics.get("r2", None),
                                "MAE": metrics.get("mae", None),
                                "RMSE": metrics.get("rmse", None),
                            })
                         # I will save this summary_df in case I need it later
                        summary_df = pd.DataFrame(summary_data)

                        # round and convert to string to force formatting
                        summary_df_table = summary_df.applymap(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
                       
                        st.write("#### Performance Summary")
                        st.table(summary_df_table.set_index("Model"))

                        # Advanced Metrics (extra details if available)
                        with st.expander("🔍 Advanced Metrics"):
                            for model, metrics in output["results"].items():
                                st.markdown(f"**{model}**")
                                for k, v in metrics.items():
                                    if isinstance(v, float):
                                        st.write(f"- **{k.upper()}**: {v:.3f}")

                        # Metric Definitions
                        with st.expander("📖 Metric Definitions"):
                            st.markdown("""
                            - **MSE (Mean Squared Error)**: The average of the squared differences between predicted and actual values.
                            - **RMSE (Root Mean Squared Error)**: Square root of the mean squared differences, penalizes large errors more.
                            - **MAE (Mean Absolute Error)**: Average absolute difference between predictions and actual values.            
                            - **R² (Coefficient of Determination)**: Proportion of variance explained by the model (1.0 = perfect).
                            """)

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