# This script helps identify the most important features that contribute to the output, i.e., payload. With this, we can 
# tailor our inputs when training different models to get maximum efficiency and minimize errors.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
import os

# Globalising all pathways
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.csv")

# Read the CSV file
df = pd.read_csv(SYNTHETIC_FILE)

# This helps produce the 'same' random values everytime.
np.random.seed(42)

# Since the excel file has the rocket data per column instead of in one row, we need to transpose the data.
df_transposed = df.set_index(['Stage', 'Parameter']).T
df_transposed.reset_index(inplace=True)
df_transposed.rename(columns={'index': 'Rocket'}, inplace=True)

# Separate features and targets
target_columns = [('Payload(kg)', 'LEO'), ('Payload(kg)', 'ISS'), ('Payload(kg)', 'SSO'), ('Payload(kg)', 'MEO'), ('Payload(kg)', 'GEO')]
y = df_transposed[target_columns].copy()
y.columns = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
X = df_transposed.drop(columns=target_columns, errors='ignore')

# Identify rockets with and without transfer stage to streamline predictions into two different categories.
has_transfer = ~X[('Transfer Stage', 'Delta-v (m/s)')].isna()

# Split data
X_with_transfer = X[has_transfer]
y_with_transfer = y[has_transfer]
X_without_transfer = X[~has_transfer]
y_without_transfer = y[~has_transfer]

# Function to preprocess the data
def preprocess_data(X):
    X_processed = X.copy()
    X_processed.columns = [f"{stage}_{param}" for stage, param in X_processed.columns]
    X_processed = X_processed.fillna(0)
    for col in X_processed.columns:
        X_processed[col] = pd.to_numeric(X_processed[col], errors='coerce')
    X_processed = X_processed.fillna(0)
    return X_processed

# Preprocess both datasets
X_with_transfer_processed = preprocess_data(X_with_transfer)
X_without_transfer_processed = preprocess_data(X_without_transfer)

# Function to get feature importance
def get_feature_importance(X, y, model, model_name, dataset_name):
    # Train the model on all data
    model.fit(X, y)
    
    # Get feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).mean(axis=0)
    else:
        # For multi-output models, get the first estimator
        if hasattr(model, 'estimators_'):
            importances = np.mean([est.feature_importances_ for est in model.estimators_], axis=0)
        else:
            print(f"Cannot extract feature importance for {model_name}")
            return None
    
    # Create importance DataFrame
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': importances
    }).sort_values('importance', ascending=False).head(10)
    
    return importance_df

# Initialize models
models = {
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(random_state=42)),
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(random_state=42))
}

# Analyze feature importance for both datasets
for dataset_name, X_data, y_data in [
    ("With Transfer Stage", X_with_transfer_processed, y_with_transfer),
    ("Without Transfer Stage", X_without_transfer_processed, y_without_transfer)
]:
    if len(X_data) > 0:
        print(f"\n{'='*60}")
        print(f"FEATURE IMPORTANCE ANALYSIS: {dataset_name}")
        print(f"{'='*60}")
        
        fig, axes = plt.subplots(1, 3, figsize=(20, 6))
        fig.suptitle(f'Top 10 Feature Importance - {dataset_name}', fontsize=16)
        
        for idx, (model_name, model) in enumerate(models.items()):
            importance_df = get_feature_importance(X_data, y_data, model, model_name, dataset_name)
            
            if importance_df is not None:
                print(f"\n{model_name} - Top 10 Features for {dataset_name}:")
                print(importance_df.to_string(index=False))
                
                # Plot feature importance
                ax = axes[idx]
                sns.barplot(x='importance', y='feature', data=importance_df, ax=ax)
                ax.set_title(f'{model_name}')
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

print("\nFeature importance analysis completed!")