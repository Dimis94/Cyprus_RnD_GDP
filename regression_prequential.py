#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np

# Define file paths for both feature selection methods
datasets = {
    "Government": ["selected_features_Government_outlier_based.csv", "selected_features_Government_hierarchical.csv"],
    "Higher_Education": ["selected_features_Higher_Education_outlier_based.csv", "selected_features_Higher_Education_hierarchical.csv"],
    "Non_Profit": ["selected_features_Non_Profit_outlier_based.csv", "selected_features_Non_Profit_hierarchical.csv"],
    "Business_Enterprises": ["selected_features_Business_Enterprises_outlier_based.csv", "selected_features_Business_Enterprises_hierarchical.csv"]
}

# Prequential Evaluation Function
def prequential_evaluation(data, target_column="GDP_percentage_change"):
    # Initialize lists to store evaluation metrics
    mae_scores, mse_scores, smape_scores = [], [], []
    coefficients = []

    # Loop through each point in time after the initial training size
    for i in range(1, len(data)):
        train = data[:i]
        test = data[i:i + 1]

        X_train = train.drop(columns=[target_column, "Year"])
        y_train = train[target_column]
        X_test = test.drop(columns=[target_column, "Year"])
        y_test = test[target_column]

        # Fit the Ridge regression model
        model = Ridge(alpha=1.0)
        model.fit(X_train, y_train)

        # Save coefficients after the last iteration
        if i == len(data) - 1:
            coefficients = model.coef_

        # Make predictions
        y_pred = model.predict(X_test)

        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / (np.abs(y_test) + np.abs(y_pred)))

        # Append metrics to the lists
        mae_scores.append(mae)
        mse_scores.append(mse)
        smape_scores.append(smape)

    # Calculate average metrics over all steps
    avg_mae = np.mean(mae_scores)
    avg_mse = np.mean(mse_scores)
    avg_smape = np.mean(smape_scores)

    return avg_mae, avg_mse, avg_smape, coefficients

# Loop through each sector and both feature-selected datasets
for sector, paths in datasets.items():
    print(f"\nProcessing {sector} Sector...")

    for method, path in zip(["Outlier-Based", "Hierarchical"], paths):
        data = pd.read_csv(path)

        # Prequential evaluation
        avg_mae, avg_mse, avg_smape, coefficients = prequential_evaluation(data)

        # Extract feature names
        feature_names = data.drop(columns=["GDP_percentage_change", "Year"]).columns

        # Create a DataFrame for coefficients
        coef_df = pd.DataFrame({
            "Feature": feature_names,
            "Coefficient": coefficients
        })

        # Save coefficients to CSV
        output_path = f"prequential_coefficients_{sector}_{method}.csv"
        coef_df.to_csv(output_path, index=False)

        # Print results
        print(f"Results for {sector} sector using {method} feature selection:")
        print(f"  - Average MAE: {avg_mae:.2f}")
        print(f"  - Average MSE: {avg_mse:.2f}")
        print(f"  - Average SMAPE: {avg_smape:.2f}%")
        print(f"  - Coefficients saved to {output_path}")

