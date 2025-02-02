#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Define datasets
datasets = {
    "Government": "selected_features_Government_hierarchical.csv",
    "Higher_Education": "selected_features_Higher_Education_hierarchical.csv",
    "Non_Profit": "selected_features_Non_Profit_hierarchical.csv",
    "Business_Enterprises": "selected_features_Business_Enterprises_hierarchical.csv",
}

def create_features(data, target_column="GDP_percentage_change", lags=2):
    """Create features with numerical stability checks"""
    feature_dfs = []
    base_data = data.copy()
    
    # Select features, excluding Year and target
    feature_cols = [col for col in data.columns if col not in ["Year", target_column]]
    
    # Create lagged features
    for lag in range(1, lags + 1):
        lagged = data[feature_cols].shift(lag)
        lagged.columns = [f"{col}_lag{lag}" for col in feature_cols]
        feature_dfs.append(lagged)
    
    # Create rolling means (more stable than other features)
    roll_mean = data[feature_cols].rolling(window=3, min_periods=1).mean()
    roll_mean.columns = [f"rolling_mean_{col}" for col in feature_cols]
    feature_dfs.append(roll_mean)
    
    # Combine features
    all_features = pd.concat([base_data] + feature_dfs, axis=1)
    
    # Replace infinities and handle missing values
    all_features = all_features.replace([np.inf, -np.inf], np.nan)
    all_features = all_features.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    return all_features

def prepare_data(data):
    """Prepare data with numerical stability checks"""
    # Handle missing and infinite values
    data = data.replace([np.inf, -np.inf], np.nan)
    data = data.fillna(method='ffill').fillna(method='bfill').fillna(0)
    
    # Remove constant columns
    constant_cols = [col for col in data.columns if data[col].nunique() == 1]
    if constant_cols:
        data = data.drop(columns=constant_cols)
    
    # Handle outliers using capping
    numeric_cols = data.select_dtypes(include=[np.number]).columns
    for col in numeric_cols:
        q1, q3 = data[col].quantile([0.01, 0.99])
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        data[col] = data[col].clip(lower=lower_bound, upper=upper_bound)
    
    return data

def safe_scale(scaler, data, fit=True):
    """Scale data with numerical stability checks"""
    if fit:
        try:
            scaled_data = scaler.fit_transform(data)
        except Exception as e:
            print(f"Warning: Scaling error - {str(e)}")
            # Fallback to simple normalization
            data_std = np.std(data, axis=0)
            data_std[data_std == 0] = 1  # Prevent division by zero
            scaled_data = (data - np.mean(data, axis=0)) / data_std
        return scaled_data
    else:
        try:
            return scaler.transform(data)
        except Exception as e:
            print(f"Warning: Scaling error - {str(e)}")
            return data

def prequential_evaluation(data, target_column="GDP_percentage_change", min_train_size=5):
    """Perform prequential evaluation with stability checks"""
    mae_scores, mse_scores, smape_scores = [], [], []
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()
    
    # Fixed alpha for stability
    model = Ridge(alpha=1.0)
    
    for i in range(min_train_size, len(data)):
        try:
            # Split data
            train = data[:i]
            test = data[i:i+1]
            
            # Prepare features
            X_train = train.drop(columns=[target_column, "Year"])
            y_train = train[target_column]
            X_test = test.drop(columns=[target_column, "Year"])
            y_test = test[target_column]
            
            # Check for valid data
            if X_train.isnull().any().any() or y_train.isnull().any():
                print(f"Warning: Iteration {i} contains null values, skipping...")
                continue
                
            # Scale features with safety checks
            X_train_scaled = safe_scale(feature_scaler, X_train)
            X_test_scaled = safe_scale(feature_scaler, X_test, fit=False)
            y_train_array = y_train.values.reshape(-1, 1)
            y_train_scaled = safe_scale(target_scaler, y_train_array).ravel()
            
            # Fit model
            model.fit(X_train_scaled, y_train_scaled)
            
            # Make predictions
            y_pred_scaled = model.predict(X_test_scaled)
            y_pred = target_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
            
            # Calculate metrics with safety checks
            mae = mean_absolute_error(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            
            # Calculate SMAPE with protection against edge cases
            eps = 1e-8
            denominator = np.abs(y_test) + np.abs(y_pred) + eps
            smape = 100 * np.mean(2 * np.abs(y_test - y_pred) / denominator)
            
            # Check for valid metrics
            if not (np.isnan(mae) or np.isnan(mse) or np.isnan(smape)):
                mae_scores.append(mae)
                mse_scores.append(mse)
                smape_scores.append(smape)
            
        except Exception as e:
            print(f"Warning: Iteration {i} failed - {str(e)}")
            continue
    
    if len(mae_scores) > 0:
        return np.mean(mae_scores), np.mean(mse_scores), np.mean(smape_scores)
    else:
        return None, None, None

# Process each dataset
print("Starting analysis with numerically stable Ridge regression...")

for sector, path in datasets.items():
    print(f"\nProcessing {sector} Sector...")
    try:
        # Read and process data
        data = pd.read_csv(path)
        
        # Prepare and create features with stability checks
        data = prepare_data(data)
        data = create_features(data, lags=2)
        
        # Run evaluation
        results = prequential_evaluation(data)
        
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

