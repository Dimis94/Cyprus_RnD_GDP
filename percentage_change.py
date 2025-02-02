#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def analyze_columns_across_sectors(sectors: Dict[str, pd.DataFrame]) -> Dict:
    column_analysis = {}
    
    # Get all unique columns
    all_columns = set()
    for df in sectors.values():
        all_columns.update(df.columns)
    
    for col in all_columns:
        if col == 'Year':
            continue
            
        column_analysis[col] = {
            'zero_percentages': {},
            'total_zero_percentage': 0,
            'present_in_sectors': 0,
            'mostly_zero_sectors': 0,
            'high_zero_sectors': 0, 
            'max_zero_percentage': 0  
        }
        
        total_rows = 0
        total_zeros = 0
        
        for sector_name, df in sectors.items():
            if col in df.columns:
                column_analysis[col]['present_in_sectors'] += 1
                
                zero_pct = (df[col] == 0).mean() * 100
                column_analysis[col]['zero_percentages'][sector_name] = zero_pct
                
                column_analysis[col]['max_zero_percentage'] = max(
                    column_analysis[col]['max_zero_percentage'], 
                    zero_pct
                )
                
                total_rows += len(df)
                total_zeros += (df[col] == 0).sum()
                
                if zero_pct > 70:
                    column_analysis[col]['mostly_zero_sectors'] += 1
                if zero_pct > 40:  # New threshold for concerning levels of zeros
                    column_analysis[col]['high_zero_sectors'] += 1
        
        if total_rows > 0:
            column_analysis[col]['total_zero_percentage'] = (total_zeros / total_rows) * 100
    
    return column_analysis

def identify_columns_to_drop(column_analysis: Dict) -> Tuple[List[str], Dict]:
    columns_to_drop = []
    drop_reasons = {}
    
    for col, analysis in column_analysis.items():
        should_drop = False
        reasons = []
        
        # Criterion 1: High overall zero percentage
        if analysis['total_zero_percentage'] > 40:
            should_drop = True
            reasons.append(f"High zero percentage across sectors: {analysis['total_zero_percentage']:.1f}%")
        
        # Criterion 2: Mostly zeros in any sector
        if analysis['max_zero_percentage'] > 70:
            should_drop = True
            reasons.append(f"Very high zeros in at least one sector: {analysis['max_zero_percentage']:.1f}%")
        
        # Criterion 3: Inconsistent data quality across sectors
        if analysis['high_zero_sectors'] >= 2:
            should_drop = True
            reasons.append(f"Poor data quality in {analysis['high_zero_sectors']} sectors")
        
        if should_drop:
            columns_to_drop.append(col)
            drop_reasons[col] = reasons
    
    return columns_to_drop, drop_reasons

def process_gdp_data(gdp_df: pd.DataFrame, base_year: int = 1998) -> pd.DataFrame:
    gdp_series = gdp_df['GDP (LCU/EUR)']
    base_gdp = gdp_series[gdp_df['Year'] == base_year].iloc[0]
    gdp_pct_changes = ((gdp_series - base_gdp) / base_gdp) * 100
    
    return pd.DataFrame({
        'Year': gdp_df['Year'],
        'GDP_pct_change': gdp_pct_changes
    })

def calculate_percentage_changes(df: pd.DataFrame, base_year: int = 1998) -> pd.DataFrame:
    df_pct = pd.DataFrame({'Year': df['Year']})
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        base_val = df[df['Year'] == base_year][col].iloc[0]
        df_pct[f'{col}_pct_change'] = ((df[col] - base_val) / base_val) * 100
    
    return df_pct
def format_output_dataframe(df_pct: pd.DataFrame, gdp_pct: pd.DataFrame) -> pd.DataFrame:
    # Rename columns to clearly indicate percentage changes
    renamed_columns = {}
    for col in df_pct.columns:
        if col != 'Year':
            renamed_columns[col] = f"{col.replace('_pct_change', '')}_percentage_change"
    
    df_final = df_pct.rename(columns=renamed_columns)
    
    # Add GDP data
    df_final = df_final.merge(gdp_pct.rename(
        columns={'GDP_pct_change': 'GDP_percentage_change'}), 
        on='Year', how='left')
    
    # Ensure consistent column ordering
    cols = ['Year'] + [col for col in df_final.columns if col != 'Year']
    df_final = df_final[cols]
    
    return df_final

def main():
    # Read data files
    sectors = {
        'Business Enterprises': pd.read_csv('business_enterprises_cleaned.csv'),
        'Government': pd.read_csv('government_cleaned.csv'),
        'Higher Education': pd.read_csv('higher_education_cleaned.csv'),
        'Private Non-Profit': pd.read_csv('private_non-profit_cleaned.csv')
    }
    gdp_df = pd.read_csv('cyprus_gdp_1998_2017_lcu.csv')
    
    # Analyze and clean columns
    print("Analyzing columns across sectors...")
    column_analysis = analyze_columns_across_sectors(sectors)
    columns_to_drop, drop_reasons = identify_columns_to_drop(column_analysis)
    
    # Print columns being removed
    print("\nColumns to be removed:")
    for col in columns_to_drop:
        print(f"\n{col}:")
        print("Reasons:")
        for reason in drop_reasons[col]:
            print(f"  - {reason}")
        print("Zero percentages by sector:")
        for sector, pct in column_analysis[col]['zero_percentages'].items():
            print(f"  {sector}: {pct:.1f}%")
    
    # Process GDP data
    gdp_pct = process_gdp_data(gdp_df)
    
    # Process each sector
    for sector_name, df in sectors.items():
        print(f"\nProcessing {sector_name}")
        
        # Remove problematic columns
        cols_to_drop = [col for col in columns_to_drop if col in df.columns]
        df_cleaned = df.drop(columns=cols_to_drop)
        
        # Calculate percentage changes
        df_pct = calculate_percentage_changes(df_cleaned)
        
        # Format final output
        df_final = format_output_dataframe(df_pct, gdp_pct)
        
        # Print summary
        print(f"\nFinal output for {sector_name}:")
        print(f"Number of metrics: {len(df_final.columns) - 2}")  # -2 for Year and GDP
        print("\nFirst 3 years sample:")
        sample_cols = ['Year', 'Total_percentage_change', 
                      'Researchers_percentage_change',
                      'Total Budget_percentage_change',
                      'GDP_percentage_change']
        print(df_final[sample_cols].head(3).to_string(index=False))
        
        # Save final output
        output_filename = f"{sector_name.lower().replace(' ', '_')}_percentage_changes.csv"
        df_final.to_csv(output_filename, index=False)
        print(f"\nSaved to {output_filename}")
        
        # Verify file contents
        verification = pd.read_csv(output_filename)
        print(f"Verification: {len(verification.columns)} columns, {len(verification)} rows")
        print("Columns:", ', '.join(verification.columns[:5]) + f"... and {len(verification.columns)-5} more")

if __name__ == "__main__":
    main()

