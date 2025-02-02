#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression

# Define datasets and sectors
datasets = {
    "Government": 'government_percentage_changes.csv',
    "Higher_Education": 'higher_education_percentage_changes.csv',
    "Non_Profit": 'private_non-profit_percentage_changes.csv',
    "Business_Enterprises": 'business_enterprises_percentage_changes.csv'
}

# Parameters
target_column = 'GDP_percentage_change'
num_top_features = 10 

# Loop through each sector dataset
for sector, path in datasets.items():
    print(f"\nProcessing {sector} Sector...")

    # Load data
    data = pd.read_csv(path)
    relevant_columns = [col for col in data.columns if col != target_column]
    X = data[relevant_columns].copy()
    y = data[target_column].copy()

    # Set first row of all features and target to zero to maintain baseline
    X.iloc[0, :] = 0
    y.iloc[0] = 0

    # Step 1: Standardize and Apply PCA (keeping 95% variance)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=0.95, random_state=42)
    X_pca = pca.fit_transform(X_scaled)

    # Adjust number of components if needed
    k_value = min(num_top_features, X_pca.shape[1])

    # Step 2: ANOVA Feature Selection on PCA Components
    selector = SelectKBest(score_func=f_regression, k=k_value)
    X_selected = selector.fit_transform(X_pca, y)
    selected_components = selector.get_support(indices=True)

    # Evaluation: Explained Variance and ANOVA F-scores
    explained_variance = sum(pca.explained_variance_ratio_[selected_components])
    f_scores = selector.scores_[selector.get_support()]

    # Output Results
    print(f"Selected PCA components for {sector}: {selected_components}")
    print(f"Explained variance of selected components: {explained_variance:.2f}")
    print(f"ANOVA F-scores of selected components: {f_scores}")

    # Save selected components with original feature names, GDP change, and Year for regression
    selected_feature_names = [relevant_columns[i] for i in selected_components]
    selected_data = data[['Year', target_column]].copy()
    selected_data[selected_feature_names] = X[selected_feature_names]
    selected_data.to_csv(f'selected_features_{sector}_hierarchical.csv', index=False)

print("\nFeature selection and initial evaluation completed for all sectors.")

