import os
from pathlib import Path
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import src.utils as utils
import src.eda as eda
import src.ml_models as ml
from src import data_preprocess
from src.data_preprocess import preprocess_df
from src.ml_models import train_and_evaluate
import src.llm_report as llm_report
from src.utils import df_download_buttons, fig_download_button


import matplotlib as mpl
from matplotlib import font_manager as fm

fam = mpl.rcParams.get("font.family", ["sans-serif"])
print("Font family set to:", fam)
resolved = fm.FontProperties(family=fam).get_name()
print("Matplotlib actually using:", resolved)


# Set page configuration
st.set_page_config(page_title="AI Data Insight Dashboard", layout="wide")
st.title("AI Data Insight Dashboard")

# --- Sidebar: Global Settings ---
with st.sidebar:
    # st.header("Global Settings")
    # st.session_state["compact_mode"] = st.checkbox("Compact mode (smaller plots, tighter layout)", value=False)
    # st.session_state["collapse_plots"] = st.checkbox("Collapse plots by default", value=False)
    # st.session_state["global_seed"] = st.number_input("Random seed", min_value=0, value=42, step=1)
    # st.session_state["cv_folds"] = st.number_input("CV folds", min_value=2, max_value=10, value=5, step=1)
    utils.sidebar_global_settings()
    utils.sidebar_llm_settings()

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
            st.session_state.dataset_name = Path(uploaded_file.name).stem
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
        st.dataframe(df.head(10))

    with tab2:
        st.subheader("Exploratory Data Analysis")

        eda.show_schema_panel(df)

        st.markdown("---")  
        eda.show_summary(df)
        st.markdown("---")
        eda.show_missing(df)
        st.markdown("---")
        eda.show_correlation(df)
        st.markdown("---")
        eda.plot_distributions(df)
        st.markdown("---")
        eda.plot_categorical(df)      
        st.markdown("---")
        eda.show_boxplot(df)
        st.markdown("---")
        eda.show_value_counts(df)
        st.markdown("---")
        eda.show_data_quality_warnings(df, target=st.session_state.get("ml_target"))

    with tab3:
        st.subheader("Machine Learning")

        if df is not None:
            target = st.selectbox("Select target variable", df.columns)
            # the target was saved when selected in the ML tab to be used in Mutual Information section of the EDA. Not being used right now.
            st.session_state["ml_target"] = target

            # the option to exclude columns from modeling (e.g. IDs, free text, obvious leakage)
            all_features = [c for c in df.columns if c != target]
            exclude_cols = st.multiselect("Exclude columns from modeling (e.g., IDs, free text, obvious leakage)",
                                          options=all_features, 
                                          key="ml_exclude_cols")

            if target:
                try:
                    # Preprocess (safe step: drops high-cardinality, splits X/y)
                    X, y = data_preprocess.preprocess_df(df, target, exclude=exclude_cols)
                    st.session_state["ml_excluded_cols"] = exclude_cols

                    # toggle to choose to use k fold cross validation
                    use_cv = st.checkbox("Use 5-fold cross-validation (metrics only)", value=False, key="ml_use_cv")

                    if use_cv:
                        cv_out = ml.cross_validate_models(X, y, cv_splits=5)
                        task_type = cv_out["task_type"]
                        st.write(f"### Detected Task: {task_type.capitalize()} (5-fold CV)")

                        if task_type == "classification":
                            rows = []
                            for model, m in cv_out["results"].items():
                                rows.append({
                                    "Model": model,
                                    "Accuracy (mean±std)": f"{m['accuracy_mean']:.3f} ± {m['accuracy_std']:.3f}",
                                    "F1 weighted (mean±std)": f"{m['f1_mean']:.3f} ± {m['f1_std']:.3f}",
                                    "ROC AUC (mean±std)": ("—" if m["roc_auc_mean"] is None else f"{m['roc_auc_mean']:.3f} ± {m['roc_auc_std']:.3f}")
                                })
                            cv_df = pd.DataFrame(rows).set_index("Model")
                            st.table(cv_df)
                            df_download_buttons("cv-metrics", cv_df, base=st.session_state.get("dataset_name", "dataset"))


                            

                        else:  # regression
                            rows = []
                            for model, m in cv_out["results"].items():
                                rows.append({
                                    "Model": model,
                                    "R² (mean±std)": f"{m['r2_mean']:.3f} ± {m['r2_std']:.3f}",
                                    "MAE (mean±std)": f"{m['mae_mean']:.3f} ± {m['mae_std']:.3f}",
                                    "RMSE (mean±std)": f"{m['rmse_mean']:.3f} ± {m['rmse_std']:.3f}",
                                })
                            cv_df = pd.DataFrame(rows).set_index("Model")
                            st.table(cv_df)
                            df_download_buttons("cv-metrics", cv_df, base=st.session_state.get("dataset_name", "dataset"))



                        st.info("Cross-validation shows typical performance across folds. Turn off the checkbox to view the single 80/20 split and diagnostics plots.")

                    else:
                        # Train models + evaluate
                        output = ml.train_and_evaluate(X, y, target)
                        task_type = output["task_type"]
                        y_test = output["y_test"]

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
                            summary_df_table = summary_df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)

                            st.write("#### Performance Summary")
                            st.table(summary_df_table.set_index("Model"))
                            df_download_buttons("test-metrics", summary_df, base=st.session_state.get("dataset_name", "dataset"))
                            
                            # Advanced Metrics
                            with st.expander("🔍 Advanced Metrics (per-class details)"):
                                st.markdown("Note: `0`, `1`, etc. in **Class/Avg** are your target classes; rows like **macro avg**/**weighted avg** are aggregates.")
                                for model, metrics in output["results"].items():
                                    st.markdown(f"**{model}**")
                                    if "classification_report" in metrics and isinstance(metrics["classification_report"], dict):
                                        report_df = pd.DataFrame(metrics["classification_report"]).transpose()
                                        report_df = report_df.round(3)
                                        st.dataframe(report_df)
                                        df_download_buttons(f"{model}-classification-report", report_df, base=st.session_state.get("dataset_name", "dataset"))
                                    else:
                                        st.write("No detailed report available.")
                            
                            # Feature Importance (Random Forest Only)
                            with st.expander("🌳 Feature Importances (Random Forest)"):
                                for model, metrics in output["results"].items():
                                    if "feature_importances" in metrics and metrics["feature_importances"]:
                                        fig = utils.plot_feature_importances(metrics["feature_importances"])
                                        if fig:
                                            st.markdown(f"**{model}**")
                                            st.pyplot(fig)
                                            fig_download_button(f"{model}-rf-importances", fig, base=st.session_state.get("dataset_name", "dataset"))


                            # Model Diagnostics
                            with st.expander("📊 Model Diagnostics (Visuals)"):
                                st.caption("Visual plots to help interpret classification performance.")

                                for model, metrics in output["results"].items():
                                    if "classification_report" in metrics:  
                                        st.markdown(f"**{model}**")

                                        preds = metrics.get("preds")
                                        probs = metrics.get("probs")
                                        if preds is None:
                                            st.info("No predictions available for this model.")
                                            continue

                                        # Confusion Matrix
                                        st.caption("• Confusion Matrix: shows how many samples were correctly or incorrectly classified.")
                                        fig = utils.plot_confusion_matrix(y_test, preds, labels=sorted(y.unique()))
                                        st.pyplot(fig)
                                        fig_download_button(f"{model}-confusion-matrix", fig, base=st.session_state.get("dataset_name", "dataset"))
                                        plt.close(fig)

                                        # ROC Curve (only for binary classification and if probabilities available)
                                        if probs is not None and y.nunique() == 2:
                                            st.caption("• ROC Curve: visualizes the model's ability to separate classes; higher curve = better.")
                                            fig = utils.plot_roc_curve(y_test, probs)
                                            st.pyplot(fig)
                                            fig_download_button(f"{model}-roc-curve", fig, base=st.session_state.get("dataset_name", "dataset"))
                                            plt.close(fig)

                                        st.markdown("---")

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
                                    #"R²": metrics.get("r2", None),
                                    "R²": metrics.get("r2_score", None),
                                    "MAE": metrics.get("mae", None),
                                    "RMSE": metrics.get("rmse", None),
                                })
                            # I will save this summary_df in case I need it later
                            summary_df = pd.DataFrame(summary_data)

                            # round and convert to string to force formatting
                            summary_df_table = summary_df.map(lambda x: f"{x:.3f}" if isinstance(x, float) else x)
                        
                            st.write("#### Performance Summary")
                            st.table(summary_df_table.set_index("Model"))
                            df_download_buttons("test-metrics", summary_df, base=st.session_state.get("dataset_name", "dataset"))

                            # Advanced Metrics (extra details if available)
                            with st.expander("🔍 Advanced Metrics"):
                                for model, metrics in output["results"].items():
                                    st.markdown(f"**{model}**")

                                    for k, v in metrics.items():
                                        if isinstance(v, float):
                                            st.write(f"- **{k.upper()}**: {v:.3f}")

                                    adv_df = pd.DataFrame({
                                        "metric": ["R²", "MAE", "RMSE", "MSE"],
                                        "value": [
                                            metrics.get("r2_score", float("nan")),
                                            metrics.get("mae", float("nan")),
                                            metrics.get("rmse", float("nan")),
                                            metrics.get("mse", float("nan")),
                                        ],
                                    })
                                    st.dataframe(adv_df.style.format({"value": "{:.3f}"}), use_container_width=True)
                                    df_download_buttons(
                                        f"{model}-advanced-metrics",
                                        adv_df,
                                        base=st.session_state.get("dataset_name", "dataset")
                                    )



                            # Feature Importance (Random Forest Only)
                            with st.expander("🌳 Feature Importances (Random Forest)"):
                                for model, metrics in output["results"].items():
                                    if "feature_importances" in metrics and metrics["feature_importances"]:
                                        fig = utils.plot_feature_importances(metrics["feature_importances"])
                                        if fig:
                                            st.markdown(f"**{model}**")
                                            st.pyplot(fig)
                                            fig_download_button(f"{model}-rf-importances", fig, base=st.session_state.get("dataset_name", "dataset"))

                            
                            # Model Diagnostics
                            with st.expander("📊 Model Diagnostics (Visuals)"):
                                    st.caption("Visual plots to help interpret model performance.")

                                    for model, metrics in output["results"].items():
                                        if "r2_score" in metrics:  
                                            st.markdown(f"**{model}**")

                                            preds = metrics.get("preds")
                                            if preds is None:
                                                st.info("No predictions available for this model.")
                                                continue

                                            figs = utils.plot_regression_diagnostics(y_test, preds)

                                            st.caption("• Residuals vs Fitted: errors should be scattered randomly if the model fits well.")
                                            st.pyplot(figs[0]) 
                                            fig_download_button(f"{model}-residuals-vs-fitted", figs[0], base=st.session_state.get("dataset_name", "dataset"))
                                            plt.close(figs[0])

                                            st.caption("• Prediction Error Plot: closer points to the diagonal line mean better predictions.")
                                            st.pyplot(figs[1])
                                            fig_download_button(f"{model}-prediction-error", figs[1], base=st.session_state.get("dataset_name", "dataset"))
                                            plt.close(figs[1])
                                            st.markdown("---")

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
        llm_report.render_llm_tab(
            df,
            default_name=st.session_state.get("dataset_name", "Dataset")
        )
        excluded_cols = st.session_state.get("ml_excluded_cols", [])
        if excluded_cols:
            st.caption(f"⚠️ The following columns were excluded from modeling: {', '.join(excluded_cols)}")

# --- Reset Button Section ---
# Add reset button at the bottom, only if a file has been processed
if st.session_state.df is not None:
    st.markdown("---") # Add a separator before the reset button for better UI
    if st.button("Reset Application"):
        reset_app()
        st.rerun() # Rerun the app to reflect the reset state

#streamlit run ai-data-dashboard/app.py