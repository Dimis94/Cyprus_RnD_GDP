#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.feature_selection import SelectKBest, mutual_info_regression
from sklearn.preprocessing import StandardScaler

# Define datasets and sectors
datasets = {
    "Government": 'government_percentage_changes.csv',
    "Higher_Education": 'higher_education_percentage_changes.csv',
    "Non_Profit": 'private_non-profit_percentage_changes.csv',
    "Business_Enterprises": 'business_enterprises_percentage_changes.csv'
}

# Parameters
target_column = 'GDP_percentage_change'
num_features = 10  

for sector, path in datasets.items():
    print(f"Processing {sector} Sector...")

    # Load data
    data = pd.read_csv(path)
    relevant_columns = [col for col in data.columns if col != target_column]

    # Step 1: Outlier Detection to Identify Informative Instances
    isolation_forest = IsolationForest(contamination=0.1, random_state=42)
    outliers = isolation_forest.fit_predict(data[relevant_columns].values)
    data_filtered = data[outliers == 1].copy()  # Retain only inliers

    # Ensure the first and last rows (1998 and 2017) are included
    if data.iloc[0].name not in data_filtered.index:
        data_filtered = pd.concat([data.iloc[[0]], data_filtered])
    if data.iloc[-1].name not in data_filtered.index:
        data_filtered = pd.concat([data_filtered, data.iloc[[-1]]])

    # Step 2: Feature Selection with Mutual Information (mutual_info_regression)
    X_filtered = data_filtered[relevant_columns]
    y_filtered = data_filtered[target_column].fillna(0)  # Ensure first row is set to zero if null

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_filtered)

    # Select top features
    selector = SelectKBest(score_func=mutual_info_regression, k=num_features)
    X_selected = selector.fit_transform(X_scaled, y_filtered)
    selected_features = X_filtered.columns[selector.get_support()]

    # Print selected features and Mutual Information scores
    print(f"Selected features for {sector}:", selected_features)
    print("Mutual Information scores of selected features:", selector.scores_[selector.get_support()])

    # Reindex to original data for consistency
    selected_data = pd.DataFrame(index=data.index)
    selected_data['Year'] = data['Year']
    selected_data[target_column] = data[target_column].fillna(0)  # Set first row of GDP change to zero

    # Add selected feature columns from filtered data
    for feature in selected_features:
        selected_data[feature] = data[feature]

    # Save to CSV
    selected_data.to_csv(f'selected_features_{sector}_outlier_based.csv', index=False)

print("\nOutlier-based feature selection and initial evaluation completed for all sectors.")

