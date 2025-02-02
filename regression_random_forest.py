#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

# Define datasets and sectors
datasets = {
    "Government": ('selected_features_Government_outlier_based.csv', 'selected_features_Government_hierarchical.csv'),
    "Higher_Education": ('selected_features_Higher_Education_outlier_based.csv', 'selected_features_Higher_Education_hierarchical.csv'),
    "Non_Profit": ('selected_features_Non_Profit_outlier_based.csv', 'selected_features_Non_Profit_hierarchical.csv'),
    "Business_Enterprises": ('selected_features_Business_Enterprises_outlier_based.csv', 'selected_features_Business_Enterprises_hierarchical.csv')
}

# Parameters
target_column = 'GDP_percentage_change'

# Cross-validation setup
tscv = TimeSeriesSplit(n_splits=5)

# Loop through each sector dataset and feature selection method
for sector, (outlier_path, hierarchical_path) in datasets.items():
    print(f"\nProcessing {sector} Sector...")

    for method, path in zip(['Outlier-Based', 'Hierarchical'], [outlier_path, hierarchical_path]):
        # Load data
        data = pd.read_csv(path)
        X = data.drop(columns=[target_column, 'Year'])
        y = data[target_column]

        # Initialize Random Forest Regressor
        rf_model = RandomForestRegressor(n_estimators=100, random_state=42)

        # Perform cross-validation and calculate metrics
        mae_scores, mse_scores, smape_scores = [], [], []
        feature_importances = None

        for train_index, test_index in tscv.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]

            rf_model.fit(X_train, y_train)
            y_pred = rf_model.predict(X_test)

            mae_scores.append(mean_absolute_error(y_test, y_pred))
            mse_scores.append(mean_squared_error(y_test, y_pred))
            smape_scores.append(100 * (abs(y_test - y_pred) / ((abs(y_test) + abs(y_pred)) / 2)).mean())

        # Save feature importances after the final iteration
        feature_importances = rf_model.feature_importances_

        # Calculate average scores
        avg_mae = sum(mae_scores) / len(mae_scores)
        avg_mse = sum(mse_scores) / len(mse_scores)
        avg_smape = sum(smape_scores) / len(smape_scores)

        # Save feature importances to a CSV file
        feature_importance_df = pd.DataFrame({
            "Feature": X.columns,
            "Importance": feature_importances
        }).sort_values(by="Importance", ascending=False)

        output_path = f"random_forest_importances_{sector}_{method}.csv"
        feature_importance_df.to_csv(output_path, index=False)

        # Output results
        print(f"Results for {sector} sector using {method} feature selection:")
        print(f"  - Average MAE: {avg_mae:.2f}")
        print(f"  - Average MSE: {avg_mse:.2f}")
        print(f"  - Average SMAPE: {avg_smape:.2f}%")
        print(f"  - Feature importances saved to {output_path}")

