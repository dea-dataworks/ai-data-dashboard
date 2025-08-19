import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, f1_score, r2_score, mean_absolute_error, mean_squared_error

def run_classification(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    # Encode categoricals
    X = pd.get_dummies(X, drop_first=True)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        st.write(f"**{name}** → Accuracy: {acc:.2f}, F1: {f1:.2f}")

        if name == "Random Forest":
            importances = model.feature_importances_
            feat_imp = pd.Series(importances, index=X.columns).sort_values(ascending=False)[:10]
            st.bar_chart(feat_imp)


def run_regression(df, target):
    X = df.drop(columns=[target])
    y = df[target]

    X = pd.get_dummies(X, drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest Regressor": RandomForestRegressor()
    }

    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        mae = mean_absolute_error(y_test, preds)
        rmse = mean_squared_error(y_test, preds, squared=False)
        st.write(f"**{name}** → R²: {r2:.2f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}")


def run_clustering(df, n_clusters=3):
    X = pd.get_dummies(df, drop_first=True)
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init="auto")
    clusters = km.fit_predict(X)
    df["Cluster"] = clusters
    st.write(df["Cluster"].value_counts())
    st.dataframe(df.head())
