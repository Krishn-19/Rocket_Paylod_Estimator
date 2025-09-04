import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import scatter_matrix

# ---------------------- Paths ----------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # assumes this script is in src/
DATA_DIR = os.path.join(BASE_DIR, "data")

REAL_FILE = os.path.join(DATA_DIR, "rockets_pivoted.xlsx")
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.xlsx")

# === Load your datasets ===
real_df = pd.read_excel(REAL_FILE)
synthetic_df = pd.read_excel(SYNTHETIC_FILE)

real_df = real_df.set_index(["Stage", "Parameter"])
synthetic_df = synthetic_df.set_index(["Stage", "Parameter"])

# === Select key parameters to compare ===
key_params = [
    ("1st Stage", "Average Isp (s)"),
    ("1st Stage", "Delta-v (m/s)"),
    ("1st Stage", "Start Mass (kg)"),
    ("1st Stage", "Final Mass (kg)"),
    ("2nd Stage", "Delta-v (m/s)"),
    ("Payload(kg)", "LEO"),
    ("Payload(kg)", "GEO")
]

real_selected = real_df.loc[key_params]

# randomly select 100 synthetic rockets
synthetic_sampled_cols = np.random.choice(synthetic_df.columns, size=100, replace=False)
synthetic_selected = synthetic_df.loc[key_params, synthetic_sampled_cols]

# === Plot side-by-side comparison ===
for param in key_params:
    plt.figure(figsize=(8,5))
    plt.plot(real_selected.columns, real_selected.loc[param], marker='o', label="Real")
    plt.plot(synthetic_selected.columns, synthetic_selected.loc[param], marker='x', linestyle='', alpha=0.7, label="Synthetic (sampled)")
    plt.xticks(rotation=45)
    plt.title(f"{param[0]} - {param[1]}")   # Stage - Parameter
    plt.ylabel(param[1])
    plt.legend()
    plt.tight_layout()
    plt.show()

# === Scatter matrix (with only sampled synthetic) ===
real_for_scatter = real_selected.T
synthetic_for_scatter = synthetic_selected.T

real_for_scatter["Dataset"] = "Real"
synthetic_for_scatter["Dataset"] = "Synthetic"

combined = pd.concat([real_for_scatter, synthetic_for_scatter])

scatter_matrix(combined.drop(columns="Dataset"), figsize=(10,10), diagonal='kde', alpha=0.7)
plt.suptitle("Scatter Matrix of Key Rocket Parameters", y=1.02)
plt.show()
