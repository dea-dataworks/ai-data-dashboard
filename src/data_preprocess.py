import pandas as pd

def preprocess_df(df: pd.DataFrame, target: str, exclude: list[str] | None = None):
    """
    Performs initial preprocessing: drops high-cardinality columns
    and splits features/target. Further preprocessing (imputation,
    encoding, scaling) happens inside the ML pipeline.

    Args:
        df (pd.DataFrame): The input DataFrame.
        target (str): The target column name.

    Returns:
        tuple: (X, y) where X is the feature DataFrame and y is the target Series.
    """
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found in DataFrame.")

    df = df.copy()

    # Drop user-specified exclusions
    if exclude:
        df.drop(columns=[c for c in exclude if c in df.columns], inplace=True)

    # Drop high-cardinality object columns (heuristic >30 unique values)
    high_card_cols = [
        col for col in df.select_dtypes(include="object").columns
        if df[col].nunique() > 30 and col != target
    ]
    if high_card_cols:
        df.drop(columns=high_card_cols, inplace=True)

    X = df.drop(columns=[target])
    y = df[target]

    return X, y
