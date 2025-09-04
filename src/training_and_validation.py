# This is the main code that trains the model and then applies those params to real data and plots for visualisation purposes.
# It first removes the outliers (data that is too far off from mean) to ensure that errors dont skyrocket.
# The inputs for the models have been selected using results from the (feature_importance_analysis.py) file.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import xgboost as xgb
from tabulate import tabulate
import os
import json
from datetime import datetime

# Loading paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(BASE_DIR, "data")
RESULTS_DIR = os.path.join(BASE_DIR, "results")

# Create results directory to save all output PNGs and CSVs
os.makedirs(RESULTS_DIR, exist_ok=True)

# Create subdirectories for better organization
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")
TABLES_DIR = os.path.join(RESULTS_DIR, "tables")
MODELS_DIR = os.path.join(RESULTS_DIR, "models")
VALIDATION_DIR = os.path.join(RESULTS_DIR, "validation")

for directory in [PLOTS_DIR, TABLES_DIR, MODELS_DIR, VALIDATION_DIR]:
    os.makedirs(directory, exist_ok=True)

SYNTHETIC_FILE = os.path.join(DATA_DIR, "synthetic_rockets_pivoted.csv")
REAL_DATA_FILE = os.path.join(DATA_DIR, "rockets_pivoted.xlsx")

# Read the CSV file
df = pd.read_csv(SYNTHETIC_FILE)

# Set random seed for reproducibility
np.random.seed(42)

# Display the structure of the data
print("Original Data Shape:", df.shape)
print("\nFirst 10 rows of the data:")
print(df.head(10))

# Transpose the data to have rockets as rows and parameters as columns
df_transposed = df.set_index(['Stage', 'Parameter']).T
df_transposed.reset_index(inplace=True)
df_transposed.rename(columns={'index': 'Rocket'}, inplace=True)

print("\nTransposed Data Shape:", df_transposed.shape)
print("\nFirst 5 rows of transposed data:")
print(df_transposed.head())

# Separate features and targets
target_columns = [('Payload(kg)', 'LEO'), ('Payload(kg)', 'ISS'), ('Payload(kg)', 'SSO'), ('Payload(kg)', 'MEO'), ('Payload(kg)', 'GEO')]
y = df_transposed[target_columns].copy()
y.columns = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
X = df_transposed.drop(columns=target_columns, errors='ignore')

# Identify rockets with and without transfer stage
has_transfer = ~X[('Transfer Stage', 'Delta-v (m/s)')].isna()
print(f"\nRockets with transfer stage: {has_transfer.sum()}")
print(f"Rockets without transfer stage: {(~has_transfer).sum()}")

# Split data into with and without transfer stage
X_with_transfer = X[has_transfer]
y_with_transfer = y[has_transfer]
X_without_transfer = X[~has_transfer]
y_without_transfer = y[~has_transfer]

print(f"\nWith transfer stage - X: {X_with_transfer.shape}, y: {y_with_transfer.shape}")
print(f"Without transfer stage - X: {X_without_transfer.shape}, y: {y_without_transfer.shape}")

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

print("\nProcessed data with transfer stage shape:", X_with_transfer_processed.shape)
print("Processed data without transfer stage shape:", X_without_transfer_processed.shape)

# Function to detect and remove outliers using IQR method
def remove_outliers_iqr(X, y, threshold=1.5):
    """
    Remove outliers using the Interquartile Range (IQR) method
    """
    # Combine X and y to ensure we remove the same rows from both
    combined = pd.concat([X, y], axis=1)
    
    # Calculate IQR for each target variable
    outliers_mask = pd.Series([False] * len(combined))
    
    for target_col in y.columns:
        Q1 = y[target_col].quantile(0.25)
        Q3 = y[target_col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        
        # Mark outliers for this target
        col_outliers = (y[target_col] < lower_bound) | (y[target_col] > upper_bound)
        outliers_mask = outliers_mask | col_outliers
    
    print(f"Number of outliers detected: {outliers_mask.sum()}")
    print(f"Percentage of data removed: {outliers_mask.sum() / len(combined) * 100:.2f}%")
    
    # Remove outliers
    clean_combined = combined[~outliers_mask]
    X_clean = clean_combined[X.columns]
    y_clean = clean_combined[y.columns]
    
    return X_clean, y_clean

# Remove outliers from both datasets
print("\nRemoving outliers from rockets with transfer stage:")
X_with_transfer_clean, y_with_transfer_clean = remove_outliers_iqr(X_with_transfer_processed, y_with_transfer)

print("\nRemoving outliers from rockets without transfer stage:")
X_without_transfer_clean, y_without_transfer_clean = remove_outliers_iqr(X_without_transfer_processed, y_without_transfer)

# Define feature sets for each model
feature_sets = {
    "Random Forest": {
        "With Transfer": [
            "Transfer Stage_Final Mass (kg)",
            "Transfer Stage_Delta-v (m/s)",
            "2nd Stage_Dry Mass (kg)",
            "1st Stage_Total Impulse (s)"
        ],
        "Without Transfer": [
            "2nd Stage_Final Mass (kg)",
            "2nd Stage_Delta-v (m/s)",
            "1st Stage_Final Mass (kg)",
            "2nd Stage_Dry Mass (kg)",
            "1st Stage_Propellant Mass (kg)",
            "1st Stage_Delta-v (m/s)",
            "1st Stage_Engine Run Time (s)"
        ]
    },
    "XGBoost": {
        "With Transfer": [
            "Transfer Stage_Final Mass (kg)",
            "Transfer Stage_Delta-v (m/s)",
            "2nd Stage_Calculation Error (m/s)",
            "1st Stage_Engine Run Time (s)",
            "2nd Stage_Dry Mass (kg)"
        ],
        "Without Transfer": [
            "2nd Stage_Final Mass (kg)",
            "2nd Stage_Delta-v (m/s)",
            "2nd Stage_Calculation Error (m/s)",
            "2nd Stage_Dry Mass (kg)",
            "1st Stage_Total Thrust (N)",
            "1st Stage_Propellant Mass (kg)"
        ]
    },
    "LightGBM": {
        "With Transfer": "all",
        "Without Transfer": "all"
    }
}

# Function to save plots with timestamp
def save_plot(fig, name, folder=PLOTS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.png"
    filepath = os.path.join(folder, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filepath}")

# Function to save tables as CSV
def save_table(df, name, folder=TABLES_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.csv"
    filepath = os.path.join(folder, filename)
    df.to_csv(filepath, index=False)
    print(f"Saved table: {filepath}")

# Function to save models
def save_model(model, name, folder=MODELS_DIR):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{name}_{timestamp}.pkl"
    filepath = os.path.join(folder, filename)
    
    # For simplicity, we'll just save the model parameters to a text file
    # In a real scenario, you might want to use joblib or pickle
    with open(filepath.replace('.pkl', '.txt'), 'w') as f:
        f.write(str(model.get_params()))
    
    print(f"Saved model info: {filepath.replace('.pkl', '.txt')}")

# Function to train and evaluate models with feature selection
def train_and_evaluate(X_train, X_test, y_train, y_test, model_name, model, feature_set_name, feature_set):
    # Select features if not using all
    if feature_set != "all":
        # Check if all requested features are available
        available_features = [f for f in feature_set if f in X_train.columns]
        missing_features = [f for f in feature_set if f not in X_train.columns]
        
        if missing_features:
            print(f"Warning: Missing features for {model_name} ({feature_set_name}): {missing_features}")
        
        X_train_selected = X_train[available_features]
        X_test_selected = X_test[available_features]
        print(f"Using {len(available_features)} features for {model_name} ({feature_set_name})")
    else:
        X_train_selected = X_train
        X_test_selected = X_test
        print(f"Using all features for {model_name} ({feature_set_name})")
    
    # Train the model
    model.fit(X_train_selected, y_train)
    
    # Make predictions on training and test data
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    
    # Calculate metrics for training data
    train_r2_scores = [r2_score(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(y_train.shape[1])]
    train_rmse_scores = [np.sqrt(mean_squared_error(y_train.iloc[:, i], y_train_pred[:, i])) for i in range(y_train.shape[1])]
    train_mae_scores = [mean_absolute_error(y_train.iloc[:, i], y_train_pred[:, i]) for i in range(y_train.shape[1])]
    
    # Calculate metrics for test data
    test_r2_scores = [r2_score(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(y_test.shape[1])]
    test_rmse_scores = [np.sqrt(mean_squared_error(y_test.iloc[:, i], y_test_pred[:, i])) for i in range(y_test.shape[1])]
    test_mae_scores = [mean_absolute_error(y_test.iloc[:, i], y_test_pred[:, i]) for i in range(y_test.shape[1])]
    
    # Calculate relative errors (RMSE as percentage of mean)
    test_rmse_percentages = [rmse / y_test.iloc[:, i].mean() * 100 for i, rmse in enumerate(test_rmse_scores)]
    
    # Create results DataFrames
    train_results = pd.DataFrame({
        'Target': y_train.columns,
        'R2': train_r2_scores,
        'RMSE': train_rmse_scores,
        'MAE': train_mae_scores
    })
    
    test_results = pd.DataFrame({
        'Target': y_test.columns,
        'R2': test_r2_scores,
        'RMSE': test_rmse_scores,
        'MAE': test_mae_scores,
        'RMSE % of Mean': test_rmse_percentages
    })
    
    # Calculate average scores
    avg_train_r2 = np.mean(train_r2_scores)
    avg_train_rmse = np.mean(train_rmse_scores)
    avg_train_mae = np.mean(train_mae_scores)
    
    avg_test_r2 = np.mean(test_r2_scores)
    avg_test_rmse = np.mean(test_rmse_scores)
    avg_test_mae = np.mean(test_mae_scores)
    avg_test_rmse_pct = np.mean(test_rmse_percentages)
    
    # Add average rows
    train_avg_row = pd.DataFrame({
        'Target': ['Average'],
        'R2': [avg_train_r2],
        'RMSE': [avg_train_rmse],
        'MAE': [avg_train_mae]
    })
    
    test_avg_row = pd.DataFrame({
        'Target': ['Average'],
        'R2': [avg_test_r2],
        'RMSE': [avg_test_rmse],
        'MAE': [avg_test_mae],
        'RMSE % of Mean': [avg_test_rmse_pct]
    })
    
    train_results = pd.concat([train_results, train_avg_row], ignore_index=True)
    test_results = pd.concat([test_results, test_avg_row], ignore_index=True)
    
    # Create plots for training data
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} - Training Data: Predicted vs Actual Values ({feature_set_name})', fontsize=16)
    
    orbits = y_train.columns
    for i, orbit in enumerate(orbits):
        row, col = i // 3, i % 3
        axes[row, col].scatter(y_train[orbit], y_train_pred[:, i], alpha=0.5)
        axes[row, col].plot([y_train[orbit].min(), y_train[orbit].max()], 
                           [y_train[orbit].min(), y_train[orbit].max()], 'r--')
        axes[row, col].set_xlabel('Actual')
        axes[row, col].set_ylabel('Predicted')
        axes[row, col].set_title(f'{orbit} Orbit')
    
    # Add text box with average metrics for training
    textstr = f'Avg R²: {avg_train_r2:.3f}\nAvg RMSE: {avg_train_rmse:.3f}\nAvg MAE: {avg_train_mae:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 2].text(0.5, 0.5, textstr, transform=axes[1, 2].transAxes, fontsize=14,
                   verticalalignment='center', horizontalalignment='center', bbox=props)
    axes[1, 2].set_axis_off()
    
    plt.tight_layout()
    save_plot(fig, f'{model_name}_{feature_set_name}_training')
    plt.show()
    
    # Create plots for test data
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f'{model_name} - Test Data: Predicted vs Actual Values ({feature_set_name})', fontsize=16)
    
    for i, orbit in enumerate(orbits):
        row, col = i // 3, i % 3
        axes[row, col].scatter(y_test[orbit], y_test_pred[:, i], alpha=0.5)
        axes[row, col].plot([y_test[orbit].min(), y_test[orbit].max()], 
                           [y_test[orbit].min(), y_test[orbit].max()], 'r--')
        axes[row, col].set_xlabel('Actual')
        axes[row, col].set_ylabel('Predicted')
        axes[row, col].set_title(f'{orbit} Orbit')
    
    # Add text box with average metrics for test
    textstr = f'Avg R²: {avg_test_r2:.3f}\nAvg RMSE: {avg_test_rmse:.3f}\nAvg MAE: {avg_test_mae:.3f}\nAvg RMSE %: {avg_test_rmse_pct:.1f}%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    axes[1, 2].text(0.5, 0.5, textstr, transform=axes[1, 2].transAxes, fontsize=14,
                   verticalalignment='center', horizontalalignment='center', bbox=props)
    axes[1, 2].set_axis_off()
    
    plt.tight_layout()
    save_plot(fig, f'{model_name}_{feature_set_name}_test')
    plt.show()
    
    # Save results to CSV
    save_table(train_results, f'{model_name}_{feature_set_name}_train_results')
    save_table(test_results, f'{model_name}_{feature_set_name}_test_results')
    
    return train_results, test_results, y_test_pred, model

# Initialize models
models = {
    'Random Forest': MultiOutputRegressor(RandomForestRegressor(n_estimators=100, random_state=42)),
    'LightGBM': MultiOutputRegressor(lgb.LGBMRegressor(random_state=42)),
    'XGBoost': MultiOutputRegressor(xgb.XGBRegressor(random_state=42))
}

# Function to run complete training and evaluation pipeline
def run_pipeline(X, y, dataset_name):
    print(f"\n{'='*50}")
    print(f"Training on {dataset_name}")
    print(f"{'='*50}")
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    all_train_results = {}
    all_test_results = {}
    all_predictions = {}
    trained_models = {}
    
    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")
        
        # Get the appropriate feature set
        feature_set = feature_sets[model_name][dataset_name]
        
        train_results, test_results, predictions, trained_model = train_and_evaluate(
            X_train, X_test, y_train, y_test, model_name, model, dataset_name, feature_set
        )
        
        all_train_results[model_name] = train_results
        all_test_results[model_name] = test_results
        all_predictions[model_name] = predictions
        trained_models[model_name] = trained_model
        
        # Save model information
        save_model(trained_model, f'{model_name}_{dataset_name}')
        
        # Print results in tabular format
        print(f"\n{model_name} Training Results for {dataset_name}:")
        print(tabulate(train_results, headers='keys', tablefmt='grid', floatfmt=".4f"))
        
        print(f"\n{model_name} Test Results for {dataset_name}:")
        print(tabulate(test_results, headers='keys', tablefmt='grid', floatfmt=".4f"))
    
    return all_train_results, all_test_results, all_predictions, trained_models, X_test, y_test

# Run pipeline for rockets with transfer stage
if len(X_with_transfer_clean) > 0:
    train_results_with_transfer, test_results_with_transfer, predictions_with_transfer, models_with_transfer, X_test_with, y_test_with = run_pipeline(
        X_with_transfer_clean, y_with_transfer_clean, "With Transfer"
    )

# Run pipeline for rockets without transfer stage
if len(X_without_transfer_clean) > 0:
    train_results_without, test_results_without, predictions_without, models_without, X_test_without, y_test_without = run_pipeline(
        X_without_transfer_clean, y_without_transfer_clean, "Without Transfer"
    )

# Create a summary table of all test results
summary_data = []

if len(X_with_transfer_clean) > 0:
    for model_name, results in test_results_with_transfer.items():
        avg_row = results[results['Target'] == 'Average'].iloc[0]
        summary_data.append([
            f"{model_name} (With Transfer)",
            avg_row['R2'],
            avg_row['RMSE'],
            avg_row['MAE'],
            avg_row['RMSE % of Mean']
        ])

if len(X_without_transfer_clean) > 0:
    for model_name, results in test_results_without.items():
        avg_row = results[results['Target'] == 'Average'].iloc[0]
        summary_data.append([
            f"{model_name} (Without Transfer)",
            avg_row['R2'],
            avg_row['RMSE'],
            avg_row['MAE'],
            avg_row['RMSE % of Mean']
        ])

# Create summary DataFrame
summary_df = pd.DataFrame(summary_data, columns=['Model', 'R2', 'RMSE', 'MAE', 'RMSE % of Mean'])
save_table(summary_df, 'model_comparison_summary')

print("\n" + "="*80)
print("SUMMARY OF ALL MODELS (Test Results)")
print("="*80)
print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))

# Print final model parameters
print("\n" + "="*80)
print("MODEL PARAMETERS (Random Forest as example)")
print("="*80)

if len(X_with_transfer_clean) > 0:
    print("\nRandom Forest parameters for rockets with transfer stage:")
    print(models_with_transfer['Random Forest'].get_params())
    
if len(X_without_transfer_clean) > 0:
    print("\nRandom Forest parameters for rockets without transfer stage:")
    print(models_without['Random Forest'].get_params())

# VALIDATION ON REAL ROCKET DATA ---------------------------------------------------------------------------------------------------------

def load_and_preprocess_real_data(file_path):
    """
    Load and preprocess the real rocket data from Excel file
    """
    print(f"\n{'='*80}")
    print("LOADING AND PREPROCESSING REAL ROCKET DATA")
    print(f"{'='*80}")
    
    # Read Excel file
    df_real = pd.read_excel(file_path)
    print(f"Real data shape: {df_real.shape}")
    print(f"\nReal data columns: {list(df_real.columns)}")
    
    # Set Stage and Parameter as index to match synthetic data format
    df_real_indexed = df_real.set_index(['Stage', 'Parameter'])
    
    # Transpose to have rockets as rows
    df_real_transposed = df_real_indexed.T
    df_real_transposed.reset_index(inplace=True)
    df_real_transposed.rename(columns={'index': 'Rocket'}, inplace=True)
    
    print(f"Transposed real data shape: {df_real_transposed.shape}")
    print(f"Rocket names: {df_real_transposed['Rocket'].tolist()}")
    
    # Separate payload (actual) data from features
    payload_columns = [col for col in df_real_transposed.columns if 'Payload(kg)' in str(col)]
    
    # Extract actual payload values (last 5 rows in original format correspond to payload data)
    y_real_actual = df_real_transposed[payload_columns].copy()
    if len(payload_columns) == 5:  # LEO, ISS, SSO, MEO, GEO
        y_real_actual.columns = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
    
    # Features are everything except payload columns
    X_real = df_real_transposed.drop(columns=payload_columns + ['Rocket'], errors='ignore')
    
    # Handle calculation errors - replace NaN with 0
    calc_error_cols = [col for col in X_real.columns if 'Calculation Error' in str(col)]
    for col in calc_error_cols:
        X_real[col] = X_real[col].fillna(0)
    
    # Identify rockets with and without transfer stage
    transfer_col = [col for col in X_real.columns if 'Transfer Stage' in str(col) and 'Delta-v' in str(col)]
    if transfer_col:
        has_transfer_real = ~X_real[transfer_col[0]].isna()
    else:
        has_transfer_real = pd.Series([False] * len(X_real))
    
    print(f"\nReal rockets with transfer stage: {has_transfer_real.sum()}")
    print(f"Real rockets without transfer stage: {(~has_transfer_real).sum()}")
    
    return X_real, y_real_actual, has_transfer_real, df_real_transposed

def validate_on_real_data(X_real, y_real_actual, has_transfer_real, trained_models, rocket_names):
    """
    Validate trained models on real rocket data
    """
    print(f"\n{'='*80}")
    print("VALIDATION ON REAL ROCKET DATA")
    print(f"{'='*80}")
    
    # Preprocess real data
    X_real_processed = preprocess_data(X_real)
    
    # Split real data based on transfer stage
    X_real_with_transfer = X_real_processed[has_transfer_real]
    X_real_without_transfer = X_real_processed[~has_transfer_real]
    y_real_with_transfer = y_real_actual[has_transfer_real]
    y_real_without_transfer = y_real_actual[~has_transfer_real]
    
    rocket_names_with = rocket_names[has_transfer_real]
    rocket_names_without = rocket_names[~has_transfer_real]
    
    validation_results = {}
    all_predictions_real = {}
    
    # Validate models for rockets with transfer stage
    if len(X_real_with_transfer) > 0 and 'models_with_transfer' in globals():
        print(f"\nValidating models on {len(X_real_with_transfer)} rockets WITH transfer stage...")
        validation_results['With Transfer'] = validate_model_group(
            X_real_with_transfer, y_real_with_transfer, rocket_names_with,
            models_with_transfer, "With Transfer"
        )
        all_predictions_real['With Transfer'] = validation_results['With Transfer']['predictions']
    
    # Validate models for rockets without transfer stage  
    if len(X_real_without_transfer) > 0 and 'models_without' in globals():
        print(f"\nValidating models on {len(X_real_without_transfer)} rockets WITHOUT transfer stage...")
        validation_results['Without Transfer'] = validate_model_group(
            X_real_without_transfer, y_real_without_transfer, rocket_names_without,
            models_without, "Without Transfer"
        )
        all_predictions_real['Without Transfer'] = validation_results['Without Transfer']['predictions']
    
    return validation_results, all_predictions_real

def validate_model_group(X_real, y_real, rocket_names, trained_models, group_name):
    """
    Validate a group of models on real data
    """
    predictions_dict = {}
    metrics_dict = {}
    
    for model_name, trained_model in trained_models.items():
        print(f"\n  Validating {model_name} for {group_name}...")
        
        # Get feature set used during training
        feature_set = feature_sets[model_name][group_name]
        
        # Select appropriate features
        if feature_set != "all":
            available_features = [f for f in feature_set if f in X_real.columns]
            missing_features = [f for f in feature_set if f not in X_real.columns]
            
            if missing_features:
                print(f"    Warning: Missing features: {missing_features}")
            
            X_real_selected = X_real[available_features]
        else:
            X_real_selected = X_real
        
        # Make predictions
        try:
            predictions = trained_model.predict(X_real_selected)
            predictions_dict[model_name] = predictions
            
            # Calculate metrics for available data (ignore NaN values)
            model_metrics = {}
            orbit_names = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
            
            for i, orbit in enumerate(orbit_names):
                if orbit in y_real.columns:
                    # Get actual and predicted values, removing NaN entries
                    actual = y_real[orbit].values
                    pred = predictions[:, i]
                    
                    # Create mask for non-NaN values
                    valid_mask = ~np.isnan(actual) & ~np.isnan(pred)
                    
                    if valid_mask.sum() > 0:  # If we have valid data points
                        actual_valid = actual[valid_mask]
                        pred_valid = pred[valid_mask]
                        
                        # Calculate metrics
                        r2 = r2_score(actual_valid, pred_valid) if len(actual_valid) > 1 else np.nan
                        rmse = np.sqrt(mean_squared_error(actual_valid, pred_valid))
                        mae = mean_absolute_error(actual_valid, pred_valid)
                        mape = np.mean(np.abs((actual_valid - pred_valid) / actual_valid)) * 100 if np.all(actual_valid != 0) else np.nan
                        
                        model_metrics[orbit] = {
                            'R2': r2,
                            'RMSE': rmse,
                            'MAE': mae,
                            'MAPE': mape,
                            'n_samples': len(actual_valid)
                        }
                    else:
                        model_metrics[orbit] = {
                            'R2': np.nan,
                            'RMSE': np.nan,
                            'MAE': np.nan,
                            'MAPE': np.nan,
                            'n_samples': 0
                        }
            
            metrics_dict[model_name] = model_metrics
            
        except Exception as e:
            print(f"    Error predicting with {model_name}: {e}")
            predictions_dict[model_name] = None
            metrics_dict[model_name] = None
    
    return {
        'predictions': predictions_dict,
        'metrics': metrics_dict,
        'rocket_names': rocket_names,
        'actual_values': y_real
    }

def create_validation_summary_tables(validation_results):
    """
    Create comprehensive summary tables for validation results
    """
    print(f"\n{'='*80}")
    print("VALIDATION RESULTS SUMMARY")
    print(f"{'='*80}")
    
    all_summary_data = []
    
    for group_name, group_results in validation_results.items():
        if group_results and 'metrics' in group_results:
            print(f"\n{group_name.upper()} ROCKETS:")
            print("-" * 50)
            
            for model_name, model_metrics in group_results['metrics'].items():
                if model_metrics:
                    model_summary = []
                    
                    for orbit, metrics in model_metrics.items():
                        if metrics['n_samples'] > 0:
                            model_summary.append([
                                f"{model_name} ({group_name})",
                                orbit,
                                metrics['R2'],
                                metrics['RMSE'],
                                metrics['MAE'],
                                metrics['MAPE'],
                                metrics['n_samples']
                            ])
                            
                            # Add to overall summary
                            all_summary_data.append([
                                f"{model_name} ({group_name})",
                                orbit,
                                metrics['R2'],
                                metrics['RMSE'],
                                metrics['MAE'],
                                metrics['MAPE'],
                                metrics['n_samples']
                            ])
                    
                    if model_summary:
                        summary_df = pd.DataFrame(model_summary, 
                                                columns=['Model', 'Orbit', 'R²', 'RMSE', 'MAE', 'MAPE%', 'N'])
                        print(f"\n{model_name} Results:")
                        print(tabulate(summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
                        
                        # Save validation results
                        save_table(summary_df, f'{model_name}_{group_name}_validation', VALIDATION_DIR)
    
    # Create overall summary table
    if all_summary_data:
        overall_summary_df = pd.DataFrame(all_summary_data, 
                                        columns=['Model', 'Orbit', 'R²', 'RMSE', 'MAE', 'MAPE%', 'N'])
        print(f"\n{'='*80}")
        print("OVERALL VALIDATION SUMMARY")
        print(f"{'='*80}")
        print(tabulate(overall_summary_df, headers='keys', tablefmt='grid', floatfmt=".4f"))
        
        # Save overall validation summary
        save_table(overall_summary_df, 'overall_validation_summary', VALIDATION_DIR)
        
        return overall_summary_df
    
    return None

def create_validation_comparison_plots(validation_results):
    """
    Create comprehensive comparison plots for validation results
    """
    print(f"\n{'='*80}")
    print("CREATING VALIDATION COMPARISON PLOTS")
    print(f"{'='*80}")
    
    orbit_names = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
    
    for group_name, group_results in validation_results.items():
        if not group_results or 'predictions' not in group_results:
            continue
            
        rocket_names = group_results['rocket_names']
        actual_values = group_results['actual_values']
        predictions = group_results['predictions']
        
        # Create subplots for each orbit
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Validation Results - {group_name} Rockets: Predicted vs Actual Payload', fontsize=16)
        
        for i, orbit in enumerate(orbit_names):
            row, col = i // 3, i % 3
            ax = axes[row, col]
            
            if orbit in actual_values.columns:
                actual = actual_values[orbit].values
                valid_mask = ~np.isnan(actual)
                
                if valid_mask.sum() > 0:
                    actual_valid = actual[valid_mask]
                    rocket_names_valid = rocket_names[valid_mask]
                    
                    # Plot predictions from each model
                    colors = ['blue', 'red', 'green', 'orange', 'purple']
                    for j, (model_name, pred) in enumerate(predictions.items()):
                        if pred is not None:
                            pred_orbit = pred[valid_mask, i]
                            ax.scatter(actual_valid, pred_orbit, 
                                     alpha=0.7, label=model_name, color=colors[j % len(colors)])
                    
                    # Plot perfect prediction line
                    min_val, max_val = actual_valid.min(), actual_valid.max()
                    ax.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.8, label='Perfect Prediction')
                    
                    ax.set_xlabel('Actual Payload (kg)')
                    ax.set_ylabel('Predicted Payload (kg)')
                    ax.set_title(f'{orbit} Orbit')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                    
                    # Add rocket names as annotations for small datasets
                    if len(actual_valid) <= 10:
                        for k, (x, name) in enumerate(zip(actual_valid, rocket_names_valid)):
                            for model_name, pred in predictions.items():
                                if pred is not None:
                                    y = pred[list(valid_mask).index(True) + k, i]
                                    ax.annotate(name, (x, y), xytext=(5, 5), 
                                              textcoords='offset points', fontsize=8, alpha=0.7)
                                    break
                else:
                    ax.text(0.5, 0.5, 'No valid data', transform=ax.transAxes, 
                           ha='center', va='center', fontsize=12)
                    ax.set_title(f'{orbit} Orbit - No Data')
            else:
                ax.text(0.5, 0.5, 'Orbit not available', transform=ax.transAxes, 
                       ha='center', va='center', fontsize=12)
                ax.set_title(f'{orbit} Orbit - Not Available')
        
        # Remove empty subplot
        axes[1, 2].axis('off')
        
        plt.tight_layout()
        save_plot(fig, f'{group_name}_validation_comparison', VALIDATION_DIR)
        plt.show()

def create_detailed_rocket_comparison_table(validation_results):
    """
    Create detailed table comparing predicted vs actual values for each rocket
    """
    print(f"\n{'='*80}")
    print("DETAILED ROCKET-BY-ROCKET COMPARISON")
    print(f"{'='*80}")
    
    for group_name, group_results in validation_results.items():
        if not group_results or 'predictions' not in group_results:
            continue
            
        print(f"\n{group_name.upper()} ROCKETS:")
        print("-" * 70)
        
        rocket_names = group_results['rocket_names']
        actual_values = group_results['actual_values']
        predictions = group_results['predictions']
        
        orbit_names = ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']
        
        for orbit in orbit_names:
            if orbit in actual_values.columns:
                print(f"\n{orbit} ORBIT:")
                print("-" * 30)
                
                # Create comparison table for this orbit
                comparison_data = []
                
                for i, rocket_name in enumerate(rocket_names):
                    actual_val = actual_values[orbit].iloc[i]
                    
                    if not np.isnan(actual_val):
                        row_data = [rocket_name, f"{actual_val:.1f}"]
                        
                        # Add predictions from each model
                        for model_name, pred in predictions.items():
                            if pred is not None:
                                pred_val = pred[i, orbit_names.index(orbit)]
                                error = ((pred_val - actual_val) / actual_val * 100) if actual_val != 0 else np.nan
                                row_data.extend([f"{pred_val:.1f}", f"{error:+.1f}%"])
                            else:
                                row_data.extend(["N/A", "N/A"])
                        
                        comparison_data.append(row_data)
                
                if comparison_data:
                    # Create column headers
                    headers = ['Rocket', 'Actual']
                    for model_name in predictions.keys():
                        headers.extend([f'{model_name} Pred', f'{model_name} Error%'])
                    
                    comparison_df = pd.DataFrame(comparison_data, columns=headers)
                    print(tabulate(comparison_df, headers='keys', tablefmt='grid', floatfmt=".1f"))
                    
                    # Save detailed comparison
                    save_table(comparison_df, f'{group_name}_{orbit}_detailed_comparison', VALIDATION_DIR)

def create_error_analysis_plots(validation_results):
    """
    Create error analysis plots showing model performance
    """
    print(f"\n{'='*80}")
    print("CREATING ERROR ANALYSIS PLOTS")
    print(f"{'='*80}")
    
    # Collect all error data
    all_errors = {}
    all_orbits = []
    
    for group_name, group_results in validation_results.items():
        if not group_results or 'metrics' not in group_results:
            continue
            
        for model_name, model_metrics in group_results['metrics'].items():
            if not model_metrics:
                continue
                
            model_key = f"{model_name} ({group_name})"
            if model_key not in all_errors:
                all_errors[model_key] = {'RMSE': [], 'MAE': [], 'MAPE': [], 'R2': [], 'orbits': []}
            
            for orbit, metrics in model_metrics.items():
                if metrics['n_samples'] > 0:
                    all_errors[model_key]['RMSE'].append(metrics['RMSE'])
                    all_errors[model_key]['MAE'].append(metrics['MAE'])
                    all_errors[model_key]['MAPE'].append(metrics['MAPE'])
                    all_errors[model_key]['R2'].append(metrics['R2'])
                    all_errors[model_key]['orbits'].append(orbit)
                    
                    if orbit not in all_orbits:
                        all_orbits.append(orbit)
    
    if not all_errors:
        print("No error data available for plotting.")
        return
    
    # Create error comparison plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Model Performance Comparison on Real Data', fontsize=16)
    
    metrics = ['RMSE', 'MAE', 'MAPE', 'R2']
    metric_titles = ['Root Mean Square Error', 'Mean Absolute Error', 'Mean Absolute Percentage Error', 'R² Score']
    
    for i, (metric, title) in enumerate(zip(metrics, metric_titles)):
        row, col = i // 2, i % 2
        ax = axes[row, col]
        
        # Prepare data for box plot
        plot_data = []
        plot_labels = []
        
        for model_key, errors in all_errors.items():
            if errors[metric]:
                plot_data.append([x for x in errors[metric] if not np.isnan(x)])
                plot_labels.append(model_key)
        
        if plot_data:
            ax.boxplot(plot_data, labels=plot_labels)
            ax.set_title(title)
            ax.set_ylabel(metric)
            ax.tick_params(axis='x', rotation=45)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f'No {metric} data available', 
                   transform=ax.transAxes, ha='center', va='center')
            ax.set_title(title)
    
    plt.tight_layout()
    save_plot(fig, 'error_analysis_comparison', VALIDATION_DIR)
    plt.show()

# Execute validation pipeline
if __name__ == "__main__":
    try:
        # Load real rocket data
        X_real, y_real_actual, has_transfer_real, df_real_full = load_and_preprocess_real_data(REAL_DATA_FILE)
        rocket_names = df_real_full['Rocket']
        
        # Validate models on real data
        validation_results, all_predictions_real = validate_on_real_data(
            X_real, y_real_actual, has_transfer_real, 
            {'models_with_transfer': models_with_transfer if 'models_with_transfer' in locals() else {},
             'models_without': models_without if 'models_without' in locals() else {}},
            rocket_names
        )
        
        # Create summary tables
        overall_summary = create_validation_summary_tables(validation_results)
        
        # Create comparison plots
        create_validation_comparison_plots(validation_results)
        
        # Create detailed comparison tables
        create_detailed_rocket_comparison_table(validation_results)
        
        # Create error analysis plots
        create_error_analysis_plots(validation_results)
        
        print(f"\n{'='*80}")
        print("VALIDATION ANALYSIS COMPLETED SUCCESSFULLY!")
        print(f"{'='*80}")
        
        # Final insights
        if overall_summary is not None:
            print(f"\nKEY INSIGHTS:")
            print("-" * 40)
            
            # Find best performing models
            for orbit in ['LEO', 'ISS', 'SSO', 'MEO', 'GEO']:
                orbit_data = overall_summary[overall_summary['Orbit'] == orbit]
                if not orbit_data.empty:
                    best_r2 = orbit_data.loc[orbit_data['R²'].idxmax()]
                    best_rmse = orbit_data.loc[orbit_data['RMSE'].idxmin()]
                    print(f"\n{orbit} Orbit:")
                    print(f"  Best R²: {best_r2['Model']} (R² = {best_r2['R²']:.3f})")
                    print(f"  Best RMSE: {best_rmse['Model']} (RMSE = {best_rmse['RMSE']:.1f})")
        
    except FileNotFoundError:
        print(f"Error: Could not find the file {REAL_DATA_FILE}")
        print("Please ensure the rockets_pivoted.xlsx file is in the data folder.")
    except Exception as e:
        print(f"Error during validation: {e}")
        import traceback
        traceback.print_exc()

print("\nAnalysis completed successfully!")