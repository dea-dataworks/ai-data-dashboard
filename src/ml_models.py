import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score

def run_models(X, y, target):
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Detect problem type
    if y.dtype == "object" or y.nunique() < 20:   # 🔹 crude heuristic
        task_type = "classification"
    else:
        task_type = "regression"

    st.write(f"### Detected task: {task_type.capitalize()}")

    results = {}

    if task_type == "classification":
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            acc = accuracy_score(y_test, preds)
            results[name] = acc
            st.write(f"**{name}** Accuracy: {acc:.3f}")
            st.text(classification_report(y_test, preds))

    else:  # regression
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100),
        }
        for name, model in models.items():
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mse = mean_squared_error(y_test, preds)
            r2 = r2_score(y_test, preds)
            results[name] = (mse, r2)
            st.write(f"**{name}** MSE: {mse:.2f}, R²: {r2:.3f}")

    return results
