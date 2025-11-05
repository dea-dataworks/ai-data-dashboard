import pandas as pd

class EDAReport:
    """Lightweight exploratory data analysis summary for a DataFrame."""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self.report = {}

    # --- Schema / Snapshot panel ---
    def schema_panel(self) -> dict:
        """Return df shape, dtype counts and missing summary."""
        df = self.df
        info = {
            "rows": len(df),
            "cols": len(df.columns),
            "numeric": len(df.select_dtypes(include=["number"]).columns),
            "categorical": len(df.select_dtypes(include=["object", "category", "string"]).columns),
            "bool": len(df.select_dtypes(include=["bool"]).columns),
            "datetime" : len(df.select_dtypes(include=["datetime64[ns]", "datetime64[ns, UTC]"]).columns),
            "mem_mb": float(round(df.memory_usage(deep=True).sum() / (1024 ** 2), 2)),
            "missing_pct": float(round((df.isna().sum().sum() / df.size * 100) if df.size else 0.0, 2)),
        }
        self.report["schema"] = info
        return info
 
        
    def missing_cardinality(self) -> dict:
        """Return missing counts/percents and high-cardinality stats."""
        df = self.df
        if df.empty:
            section = {"missing": {"count": {}, "pct": {}}, "cardinality": {}}
            self.report["missing_cardinality"] = section
            return section

        # Missing counts and % per column (only columns with any missing)
        miss_counts = {col: int(cnt) for col, cnt in df.isnull().sum().items() if cnt > 0}
        total_rows = len(df)
        miss_pct = {col: round((cnt / total_rows) * 100, 2) for col, cnt in miss_counts.items()}

        # High-cardinality for text-like columns
        card = {
            c: int(df[c].nunique(dropna=True))
            for c in df.select_dtypes(["object", "category", "string"]).columns
        }

        section = {
            "missing": {"count": miss_counts, "pct": miss_pct},
            "cardinality": card,
        }
        # store once under one key (so you can fetch the pair together later)
        self.report["missing_cardinality"] = section
        return section

    
    def analyze_data_quality(self, target: str | None = None) -> dict:
        """Check for duplicates, identical-to-target leakage, and high correlation."""
        df = self.df
        issues = []

        # duplicates
        dup_count = int(df.duplicated().sum())
        if dup_count > 0:
            issues.append({"type": "duplicates", "count": dup_count})

        # Target leakage: identical columns
        if target and target in df.columns:
            identical_cols = [
                col for col in df.columns
                if col != target and df[col].equals(df[target])
            ]
            if identical_cols:
                issues.append({"type": "identical_to_target", "columns": identical_cols})

        # High correlation with target (numeric only)
        if target and target in df.columns and pd.api.types.is_numeric_dtype(df[target]):
            corr = df.corr(numeric_only=True)
            if target in corr.columns:
                high_corr = corr[target].drop(target).abs()
                suspicious = high_corr[high_corr >= 0.95]
                if not suspicious.empty:
                    issues.append({
                        "type": "high_corr",
                        "details": suspicious.round(3).to_dict()
                    })
        
        # Store + return
        self.report["data_quality"] = issues
        return issues


    

        
            
        

        


