import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, classification_report,
    mean_squared_error, mean_absolute_error, r2_score,
     ConfusionMatrixDisplay, RocCurveDisplay
)
from sklearn.dummy import DummyClassifier, DummyRegressor



def detect_task_type(y: pd.Series) -> str:
    """Detect if target is classification or regression."""
    if pd.api.types.is_numeric_dtype(y) and 1 < y.nunique() < 20:
        return "classification"
    elif pd.api.types.is_object_dtype(y) or pd.api.types.is_categorical_dtype(y):
        return "classification"
    elif pd.api.types.is_numeric_dtype(y):
        return "regression"
    else:
        raise ValueError("Invalid target column type.")


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Return a preprocessing pipeline for numerical + categorical features."""
    numeric_features = X.select_dtypes(include=['number']).columns
    categorical_features = X.select_dtypes(include=['object', 'category', 'bool']).columns

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])

    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='passthrough'
    )


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, target_column: str) -> dict:
    """
    Train models (classification/regression) and return evaluation results.
    """
    task_type = detect_task_type(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42,
        stratify=y if task_type == "classification" else None
    )

    preprocessor = get_preprocessor(X)

    results = {}

    if task_type == "classification":
        models = {
            "Dummy Classifier": DummyClassifier(strategy="most_frequent"),
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42),
            "Random Forest Classifier": RandomForestClassifier(n_estimators=100, random_state=42),
        }

        for name, model in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            pipe.fit(X_train, y_train)

            preds = pipe.predict(X_test)
            probs = pipe.predict_proba(X_test)[:, 1] if hasattr(pipe, "predict_proba") else None

            results[name] = {
                "accuracy": accuracy_score(y_test, preds),
                "f1_score": f1_score(y_test, preds, average="weighted"),
                "roc_auc": roc_auc_score(y_test, probs) if probs is not None and y.nunique() == 2 else None,
                "classification_report": classification_report(y_test, preds, output_dict=True),
                "classification_report_text": classification_report(y_test, preds),  # optional
                "preds": preds,
                "probs": probs,
                }

    else:  # regression
        models = {
            "Dummy Regressor": DummyRegressor(strategy="mean"),
            "Linear Regression": LinearRegression(),
            "Random Forest Regressor": RandomForestRegressor(n_estimators=100, random_state=42),
        }

        for name, model in models.items():
            pipe = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])
            pipe.fit(X_train, y_train)
            preds = pipe.predict(X_test)

            mse = mean_squared_error(y_test, preds)
            results[name] = {
                "mse": mse,
                "rmse": mse**0.5,
                "mae": mean_absolute_error(y_test, preds),
                "r2_score": r2_score(y_test, preds),
                "preds": preds
            }

    return {"task_type": task_type, "results": results, "y_test": y_test}

