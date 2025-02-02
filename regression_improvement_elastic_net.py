#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
from sklearn.linear_model import ElasticNetCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
from sklearn.exceptions import ConvergenceWarning
import warnings
warnings.filterwarnings('ignore', category=ConvergenceWarning)

# Define datasets
datasets = {
    "Government": "selected_features_Government_hierarchical.csv",
    "Higher_Education": "selected_features_Higher_Education_hierarchical.csv",
    "Non_Profit": "selected_features_Non_Profit_hierarchical.csv",
    "Business_Enterprises": "selected_features_Business_Enterprises_hierarchical.csv",
}

def prepare_data(data):
    """Prepare data by handling missing values and removing constant columns"""
    # Remove constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() == 1]
    if constant_cols:
        data = data.drop(columns=constant_cols)
    
    # Handle any missing values
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    return data

def elasticnet_prequential_optimized(data, target_column="GDP_percentage_change"):
    mae_scores, mse_scores, smape_scores = [], [], []
    
    # Prepare the data
    data = prepare_data(data)
    
    # Use RobustScaler instead of StandardScaler for better handling of outliers
    feature_scaler = RobustScaler()
    target_scaler = RobustScaler()
    
    # Get feature columns (excluding target and Year)
    feature_cols = [col for col in data.columns if col not in [target_column, "Year"]]
    
    # Scale features and target
    data[feature_cols] = feature_scaler.fit_transform(data[feature_cols])
    data[target_column] = target_scaler.fit_transform(data[[target_column]])
    
    # Minimum samples needed for stable cross-validation
    min_samples = 5
    
    for i in range(min_samples, len(data)):
        train = data[:i]
        test = data[i:i+1]
        
        X_train = train[feature_cols]
        y_train = train[target_column]
        X_test = test[feature_cols]
        y_test = test[target_column]
        
        # Optimize ElasticNet parameters
        model = ElasticNetCV(
            l1_ratio=[.1, .3, .5, .7, .9, .95, .99, 1],  # More focused range
            alphas=np.logspace(-5, 2, 30),              # Adjusted range
            cv=min(3, len(X_train)-1),                  # Smaller CV splits
            max_iter=5000,                              # Reduced iterations
            tol=1e-4,                                   # Increased tolerance
            random_state=42,
            selection='random',                         # Faster computation
            n_jobs=-1                                   # Parallel processing
        )
        
        try:
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Rescale predictions and actuals
            y_pred = target_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
            y_test = target_scaler.inverse_transform(y_test.values.reshape(-1, 1)).flatten()
            
            # Calculate metrics
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Handle edge cases in SMAPE calculation
            eps = 1e-10  # Small constant to prevent division by zero
            smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / 
                                (np.abs(y_test) + np.abs(y_pred) + eps))
            
            mae_scores.append(mae)
            mse_scores.append(mse)
            smape_scores.append(smape)
            
        except Exception as e:
            print(f"Warning: Iteration {i} failed with error: {str(e)}")
            continue
    
    # Return average metrics only if we have scores
    if len(mae_scores) > 0:
        return np.mean(mae_scores), np.mean(mse_scores), np.mean(smape_scores)
    else:
        return None, None, None

# Process each dataset
print("Starting analysis...")
for sector, path in datasets.items():
    print(f"\nProcessing {sector} Sector with Optimized ElasticNet...")
    try:
        # Read and process data
        data = pd.read_csv(path)
        
        # Run ElasticNet prequential evaluation
        results = elasticnet_prequential_optimized(data)
        
        if results[0] is not None:
            avg_mae, avg_mse, avg_smape = results
            print(f"Results for {sector}:")
            print(f"  - Average MAE: {avg_mae:.2f}")
            print(f"  - Average MSE: {avg_mse:.2f}")
            print(f"  - Average SMAPE: {avg_smape:.2f}%")
        else:
            print(f"Warning: Could not compute metrics for {sector}")
            
    except Exception as e:
        print(f"Error processing {sector}: {str(e)}")
        continue

