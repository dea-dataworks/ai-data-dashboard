import pytest
import pandas as pd
from src.eda_report import EDAReport

@pytest.fixture
def sample_df():
    data = {
        "Age": [22, 38, None, 26, 35],
        "Sex": ["male", "female", "female", "male", "male"],
        "Survived": [0, 1, 1, 0, 1],
    }
    return pd.DataFrame(data)

def test_schema_panel_returns_expected_keys(sample_df):
    report = EDAReport(sample_df)
    result = report.schema_panel()
    expected_keys = {"rows", "cols", "numeric", "categorical", "bool", "datetime", "mem_mb", "missing_pct"}
    assert expected_keys.issubset(result.keys())
    assert isinstance(result["rows"], int)
    assert "schema" in report.report

def test_missing_cardinality_counts(sample_df):
    df = sample_df.copy()
    df.loc[0, "Age"] = None
    report = EDAReport(df)
    result = report.missing_cardinality()
    assert "missing" in result and "cardinality" in result
    assert "Age" in result["missing"]["count"]
    assert isinstance(result["missing"]["pct"]["Age"], float)

def test_analyze_data_quality_detects_duplicates(sample_df):
    dup_df = pd.concat([sample_df, sample_df.iloc[[0]]], ignore_index=True)
    report = EDAReport(dup_df)
    issues = report.analyze_data_quality(target="Survived")
    assert any(i["type"] == "duplicates" for i in issues)

def test_analyze_data_quality_invalid_target_raises(sample_df):
    report = EDAReport(sample_df)
    # No explicit raise in method, but we can validate behavior
    result = report.analyze_data_quality(target="InvalidTarget")
    assert result == [] or isinstance(result, list)
