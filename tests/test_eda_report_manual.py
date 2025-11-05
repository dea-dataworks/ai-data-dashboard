import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.eda_report import EDAReport
import pandas as pd

# --- Load dataset ---
df = pd.read_csv("examples/titanic.csv")
eda = EDAReport(df)

# --- Test schema panel ---
schema = eda.schema_panel()
print("\n--- Schema Panel ---")
print(schema)

# --- Test missing + cardinality ---
mc = eda.missing_cardinality()
print("\n--- Missing & Cardinality ---")
print("Missing columns:", list(mc["missing"]["count"].keys()))
print("Cardinality columns:", list(mc["cardinality"].keys()))

# --- Test data quality ---
dq = eda.analyze_data_quality(target="Survived")
print("\n--- Data Quality ---")
if dq:
    for issue in dq:
        print(issue)
else:
    print("No major issues detected.")
