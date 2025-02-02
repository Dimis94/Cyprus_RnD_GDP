#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

# Define datasets and sectors
datasets = {
    "Government": "selected_features_Government_outlier_based.csv",
    "Higher_Education": "selected_features_Higher_Education_outlier_based.csv",
    "Non_Profit": "selected_features_Non_Profit_outlier_based.csv",
    "Business_Enterprises": "selected_features_Business_Enterprises_outlier_based.csv"
}

# Define lagged variables function
def add_lagged_features(data, target_column, lags=1):
    df = data.copy()
    for col in data.columns:
        if col != target_column:
            for lag in range(1, lags + 1):
                df[f'{col}_lag{lag}'] = df[col].shift(lag)
    df = df.dropna().reset_index(drop=True)  # Drop rows with NaN values after shifting
    return df

# Function to calculate SMAPE
def smape(y_true, y_pred):
    return 100 / len(y_true) * np.sum(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred)))

# Evaluation function
def evaluate_model(model, X, y):
    y_pred = model.predict(X)
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    smape_score = smape(y, y_pred)
    return mae, mse, smape_score

# Loop through each sector
results = {}
for sector, path in datasets.items():
    print(f"\nProcessing {sector} Sector...")

    # Load data
    data = pd.read_csv(path)
    target_column = 'GDP_percentage_change'
    
    # Add lagged features
    lagged_data = add_lagged_features(data, target_column, lags=1)  # You can adjust lags

    # Prepare training data
    X = lagged_data.drop(columns=[target_column])
    y = lagged_data[target_column]
    
    # Initialize Bayesian Ridge Regression
    model = BayesianRidge()

    # Cross-validate
    mae_scores = -cross_val_score(model, X, y, cv=5, scoring=make_scorer(mean_absolute_error))
    mse_scores = -cross_val_score(model, X, y, cv=5, scoring=make_scorer(mean_squared_error))
    smape_scores = -cross_val_score(model, X, y, cv=5, scoring=make_scorer(smape))

    # Compute average scores
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_smape = np.mean(smape_scores)

    # Display results
    results[sector] = {
        'Average MAE': avg_mae,
        'Average MSE': avg_mse,
        'Average SMAPE': avg_smape
    }
    print(f"Results for {sector} sector:")
    print(f"  - Average MAE: {avg_mae:.2f}")
    print(f"  - Average MSE: {avg_mse:.2f}")
    print(f"  - Average SMAPE: {avg_smape:.2f}%")

# Summary of results for each sector
print("\nSummary of Results with Time-Lagged Variables:")
for sector, metrics in results.items():
    print(f"{sector} sector:")
    for metric, value in metrics.items():
        print(f"  - {metric}: {value:.2f}")

