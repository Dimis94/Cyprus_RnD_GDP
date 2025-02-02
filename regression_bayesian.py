#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, make_scorer

# Define function to evaluate Bayesian Regression with MAE, MSE, SMAPE, and Coefficients
def evaluate_bayesian_regression(X, y):
    model = BayesianRidge()
    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize metrics
    mae_scores = []
    mse_scores = []
    smape_scores = []
    coefficients = np.zeros(X.shape[1])  # For storing the average coefficients

    # Perform cross-validation
    for train_idx, test_idx in cv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae_scores.append(mean_absolute_error(y_test, y_pred))
        mse_scores.append(mean_squared_error(y_test, y_pred))
        smape_scores.append(np.mean(2 * np.abs(y_pred - y_test) / (np.abs(y_pred) + np.abs(y_test)) * 100))

        # Accumulate coefficients
        coefficients += model.coef_

    # Average coefficients across folds
    coefficients /= cv.n_splits

    return np.mean(mae_scores), np.mean(mse_scores), np.mean(smape_scores), coefficients

# Define paths for datasets and target column
datasets = {
    "Government": {
        "Outlier-Based": 'selected_features_Government_outlier_based.csv',
        "Hierarchical": 'selected_features_Government_hierarchical.csv'
    },
    "Higher_Education": {
        "Outlier-Based": 'selected_features_Higher_Education_outlier_based.csv',
        "Hierarchical": 'selected_features_Higher_Education_hierarchical.csv'
    },
    "Non_Profit": {
        "Outlier-Based": 'selected_features_Non_Profit_outlier_based.csv',
        "Hierarchical": 'selected_features_Non_Profit_hierarchical.csv'
    },
    "Business_Enterprises": {
        "Outlier-Based": 'selected_features_Business_Enterprises_outlier_based.csv',
        "Hierarchical": 'selected_features_Business_Enterprises_hierarchical.csv'
    }
}
target_column = 'GDP_percentage_change'

# Process each sector and feature selection approach
for sector, paths in datasets.items():
    print(f"\nProcessing {sector} Sector...")

    for approach, path in paths.items():
        # Load data
        data = pd.read_csv(path)
        X = data.drop(columns=[target_column, 'Year'])
        y = data[target_column]

        # Standardize X to ensure comparability
        X_scaled = (X - X.mean()) / X.std()

        # Evaluate Bayesian Regression
        avg_mae, avg_mse, avg_smape, coefficients = evaluate_bayesian_regression(X_scaled, y)

        # Save coefficients to a DataFrame and export as CSV
        coef_df = pd.DataFrame({
            "Feature": X.columns,
            "Coefficient": coefficients
        }).sort_values(by="Coefficient", ascending=False)

        output_path = f"bayesian_coefficients_{sector}_{approach}.csv"
        coef_df.to_csv(output_path, index=False)

        # Display results
        print(f"\nResults for {sector} sector using {approach} feature selection:")
        print(f"  - Average MAE: {avg_mae:.2f}")
        print(f"  - Average MSE: {avg_mse:.2f}")
        print(f"  - Average SMAPE: {avg_smape:.2f}%")
        print(f"  - Coefficients saved to {output_path}")

