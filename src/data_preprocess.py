import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_df(df: pd.DataFrame, target: str):
    df = df.copy()

    # Drop high-cardinality text columns (heuristic: >30 unique values and dtype object)
    high_card_cols = [col for col in df.select_dtypes(include="object").columns 
                      if df[col].nunique() > 30 and col != target]
    df.drop(columns=high_card_cols, inplace=True)

    # Fill missing values
    for col in df.columns:
        if df[col].dtype in ["int64", "float64"]:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)

    # One-hot encode categoricals
    df = pd.get_dummies(df, drop_first=True)

    # Split features/target
    X = df.drop(columns=[target])
    y = df[target]

    # Scale numerics (optional but safer for linear models)
    scaler = StandardScaler()
    X_scaled = X.copy()
    X_scaled[X.select_dtypes(include=["int64","float64"]).columns] = scaler.fit_transform(
        X.select_dtypes(include=["int64","float64"])
    )

    return X_scaled, y
