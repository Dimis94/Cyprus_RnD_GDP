#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np

# Load the Excel file and the necessary sheets
file_path = 'R&D_Data.xlsx'

# Load the main sheets for existing data
sheets = ['I.1.2a.', 'I.1.3a.', 'I.1.5a.', 'I.1.7.', 'I.1.8.', 'I.1.9.', 'I.1.10.']
dataframes = {sheet: pd.read_excel(file_path, sheet_name=sheet, skiprows=5) for sheet in sheets}

# Function to clean column names
def clean_column_names(df):
    df.columns = df.columns.str.replace('\n', ' ', regex=False).str.strip()
    df.columns = df.columns.str.replace(r'\s+', ' ', regex=True)
    return df

# Clean column names for all dataframes
for sheet, df in dataframes.items():
    dataframes[sheet] = clean_column_names(df)

column_renames = {
    'I.1.2a.': {"Sector of Performance": "Sector of Performance", "Unnamed: 3": "Researchers", "Unnamed: 6": "Technicians", "Unnamed: 9": "Other Supporting Staff"},
    'I.1.3a.': {"Sector of Performance": "Sector of Performance", "PhD Holders": "PhD Holders", "Post-graduate Degree Holders": "Post-graduate Degree Holders", "University Degree Holders": "University Degree Holders", "Other Post Secondary Diplomas": "Other Post Secondary Diplomas", "Secondary Education": "Secondary Education", "Primary Education": "Primary Education"},
    'I.1.5a.': {"Sector of Performance": "Sector of Performance", "Natural Sciences": "Natural Sciences", "Engineering and Technology": "Engineering and Technology", "Medical Sciences": "Medical Sciences", "Agricultural Sciences": "Agricultural Sciences", "Social Sciences": "Social Sciences", "Humanities": "Humanities"},
    'I.1.7.': {"Sector of Performance": "Sector of Performance", "Labour Costs": "Labour Costs", "Capital Expenditure": "Capital Expenditure", "Other Current Expenditure": "Other Current Expenditure"},
    'I.1.8.': {"Sector of Performance": "Sector of Performance", "Basic Research": "Basic Research", "Applied Research": "Applied Research", "Experimental Development": "Experimental Development"},
    'I.1.9.': {"Sector of Performance": "Sector of Performance", "Natural Sciences": "Natural Sciences Expenditure", "Engineering and Technology": "Engineering and Technology Expenditure", "Medical Sciences": "Medical Sciences Expenditure", "Agricultural Sciences": "Agricultural Sciences Expenditure", "Social Sciences": "Social Sciences Expenditure", "Humanities": "Humanities Expenditure"},
    'I.1.10.': {"Sector of Performance": "Sector of Performance", "Government Budget": "Government Budget", "Public Universities Budget": "Public Universities Budget", "Self Financing": "Self Financing", "Business Enterprises": "Business Enterprises", "Research and Innovation Foundation": "Research and Innovation Foundation", "Private Non-profit Institutions": "Private Non-profit Institutions", "European Union": "European Union", "Other Sources from Abroad": "Other Sources from Abroad", "Total": "Total Budget"}
}

for sheet, renames in column_renames.items():
    dataframes[sheet] = dataframes[sheet].rename(columns=renames)

for df in dataframes.values():
    if 'Sector of Performance' in df.columns:
        df['Sector of Performance'] = df['Sector of Performance'].str.strip()

# Handle null values, convert numeric columns to float, and remove decimal places
def clean_numeric_values(df):
    # Replace "..." with 0
    df = df.replace('…', 0)
    
    # Handle null values
    df = df.fillna(0)
    
    # Convert numeric columns to float and remove decimal places
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].astype(float).astype(int)
    return df

for sheet, df in dataframes.items():
    dataframes[sheet] = clean_numeric_values(df)

years = list(range(1998, 2018))

def merge_sectors(dataframes, sector_name):
    filtered_dfs = []
    
    for df in dataframes.values():
        if 'Sector of Performance' not in df.columns:
            print(f"Warning: 'Sector of Performance' not found in columns: {df.columns}")
            continue
        filtered_df = df[df['Sector of Performance'].str.contains(sector_name, na=False, case=False)].reset_index(drop=True)
        filtered_dfs.append(filtered_df)
    
    # Merge the filtered dataframes
    df_merged = pd.concat(filtered_dfs, axis=1)
    
    # Remove duplicate 'Sector of Performance' columns
    df_merged = df_merged.loc[:,~df_merged.columns.duplicated()]
    
    # Drop unnecessary 'Unnamed' columns
    df_merged = df_merged.loc[:, ~df_merged.columns.str.contains('Unnamed', case=False)]
    
    # Set the index to years
    df_merged['Year'] = years
    df_merged.set_index('Year', inplace=True)
    
    return df_merged

# Handle null values and generate descriptive analysis
def handle_null_values_and_descriptive(df):
    # Count null values before handling
    null_counts = df.isnull().sum()
    total_nulls = null_counts.sum()
    
    # Handle null values
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(0)
    non_numeric_columns = df.select_dtypes(exclude=[np.number]).columns
    df[non_numeric_columns] = df[non_numeric_columns].fillna('Unknown')
    
    # Print number of null/unknown values for each column
    print(f"Null values before handling: {total_nulls}")
    print(null_counts[null_counts > 0])
    
    # Generate descriptive statistics
    descriptive_stats = df.describe().astype(int)
    return df, descriptive_stats

sectors = ["Government", "Business enterprises", "Higher education", "Private non-profit"]

for sector in sectors:
    # Merge data for each sector
    sector_df = merge_sectors(dataframes, sector)
    
    sector_df, desc_stats = handle_null_values_and_descriptive(sector_df)
    
    # Export merged data and descriptive stats to CSV
    sector_name = sector.replace(" ", "_").lower()
    sector_df.to_csv(f'{sector_name}_data_1998_2017.csv')
    desc_stats.to_csv(f'{sector_name}_descriptive_stats_1998_2017.csv')

