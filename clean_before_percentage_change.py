#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
from typing import Dict, Tuple

def categorize_columns(df: pd.DataFrame) -> Dict[str, list]:
    categories = {
        'monetary': [],
        'personnel': [],
        'research_fields': [],
        'other': []
    }
    
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        # Determine primary category
        if 'expenditure' in col.lower() or 'budget' in col.lower() or 'cost' in col.lower():
            categories['monetary'].append(col)
        elif any(term in col.lower() for term in ['researchers', 'technicians', 'staff', 'holders']):
            categories['personnel'].append(col)
        elif any(term in col.lower() for term in ['sciences', 'engineering', 'medical', 'agricultural', 'humanities']):
            if col not in categories['monetary']:  # Avoid duplication
                categories['research_fields'].append(col)
        else:
            categories['other'].append(col)
    
    return categories

def analyze_zero_patterns(df: pd.DataFrame) -> Dict[str, Dict]:
    zero_patterns = {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col == 'Year':
            continue
            
        series = df[col]
        base_year_value = series.iloc[0]  # 1998 value
        zero_mask = series == 0
        
        # Calculate consecutive zeros only if there are any zeros
        if zero_mask.any():
            consecutive_zeros = max(len(list(g)) 
                                 for _, g in pd.Series(zero_mask)
                                 .groupby(zero_mask.cumsum()) 
                                 if g.iloc[0])
        else:
            consecutive_zeros = 0
            
        zero_patterns[col] = {
            'zero_count': zero_mask.sum(),
            'zero_percentage': zero_mask.mean() * 100,
            'base_year_zero': base_year_value == 0,
            'consecutive_zeros': consecutive_zeros,
            'years_with_zeros': df.loc[zero_mask, 'Year'].tolist()
        }
    
    return zero_patterns

def get_appropriate_minimum(df: pd.DataFrame, col: str, category: str) -> float:
    non_zero_values = df[col][df[col] > 0]
    if non_zero_values.empty:
        return 1.0
    
    min_val = non_zero_values.min()
    mean_val = non_zero_values.mean()
    median_val = non_zero_values.median()
    
    if category == 'monetary':
        if mean_val > 1000:
            # For large monetary values, use 5% of median
            return median_val * 0.05
        else:
            # For smaller monetary values, use 10% of median
            return median_val * 0.10
    elif category == 'personnel':
        # For personnel, use minimum 1 or 10% of median
        return max(1.0, median_val * 0.10)
    elif category == 'research_fields':
        # For research fields, use 15% of median
        return median_val * 0.15
    else:
        # For other categories, use 20% of median if base year zero
        if df[col].iloc[0] == 0:
            return median_val * 0.20
        else:
            return min_val * 0.50

def handle_zeros(df: pd.DataFrame, zero_patterns: Dict[str, Dict], 
                categories: Dict[str, list]) -> Tuple[pd.DataFrame, Dict]:
    df_cleaned = df.copy()
    replacements = {}
    
    for category, cols in categories.items():
        for col in cols:
            pattern = zero_patterns[col]
            if pattern['zero_count'] == 0:
                continue
                
            # Skip if clearly structural zeros
            if pattern['zero_percentage'] > 75:
                continue
                
            # Get appropriate replacement value
            replacement_value = get_appropriate_minimum(df, col, category)
            
            # Special handling for base year zeros
            if pattern['base_year_zero']:
                # Use a higher percentage for base year
                base_year_replacement = df[col][df[col] > 0].median() * 0.25  # 25% of median
                replacement_value = max(replacement_value, base_year_replacement)
            
            # Track replacements
            replacements[col] = {
                'original_zeros': pattern['zero_count'],
                'category': category,
                'replacement_value': replacement_value,
                'zero_percentage': pattern['zero_percentage'],
                'base_year_zero': pattern['base_year_zero'],
                'original_stats': {
                    'min': df[col][df[col] > 0].min(),
                    'median': df[col].median(),
                    'mean': df[col].mean()
                }
            }
            
            # Apply replacement
            df_cleaned.loc[df_cleaned[col] == 0, col] = replacement_value
    
    return df_cleaned, replacements

def validate_cleaning(df_original: pd.DataFrame, df_cleaned: pd.DataFrame, 
                     replacements: Dict) -> Dict:
    validation = {
        'total_changes': 0,
        'significant_changes': [],
        'base_year_changes': []
    }
    
    for col in df_original.columns:
        if col == 'Year':
            continue
            
        if col in replacements:
            # Check total changes
            changes = (df_original[col] != df_cleaned[col]).sum()
            validation['total_changes'] += changes
            
            # Check base year
            if df_original[col].iloc[0] == 0:
                validation['base_year_changes'].append({
                    'column': col,
                    'original': 0,
                    'cleaned': df_cleaned[col].iloc[0],
                    'median': df_cleaned[col].median()
                })
            
            # Check for significant changes
            max_val = df_original[col][df_original[col] > 0].max()
            if max_val > 0:
                relative_change = replacements[col]['replacement_value'] / max_val
                if relative_change > 0.3:  # More than 30% of max
                    validation['significant_changes'].append({
                        'column': col,
                        'relative_change': relative_change,
                        'replacement': replacements[col]['replacement_value'],
                        'max_value': max_val
                    })
    
    return validation

def print_enhanced_summary(df: pd.DataFrame, df_cleaned: pd.DataFrame, 
                         categories: Dict[str, list], replacements: Dict,
                         validation: Dict):
    print("\nEnhanced Cleaning Summary:")
    print(f"Total columns processed: {len(df.columns)}")
    print(f"Total value changes: {validation['total_changes']}")
    
    if validation['base_year_changes']:
        print("\nBase Year (1998) Changes:")
        for change in validation['base_year_changes']:
            print(f"\n  {change['column']}:")
            print(f"    Original: 0")
            print(f"    Cleaned: {change['cleaned']:.3f}")
            print(f"    Column Median: {change['median']:.3f}")
    
    if validation['significant_changes']:
        print("\nSignificant Value Changes (>30% of max):")
        for change in validation['significant_changes']:
            print(f"\n  {change['column']}:")
            print(f"    Replacement: {change['replacement']:.3f}")
            print(f"    Max Value: {change['max_value']:.3f}")
            print(f"    Relative Change: {change['relative_change']*100:.1f}%")
    
    for category, cols in categories.items():
        print(f"\n{category.title()} Columns ({len(cols)}):")
        category_replacements = {col: info for col, info in replacements.items() 
                               if info['category'] == category}
        
        if category_replacements:
            print("\nColumns with zero replacements:")
            for col, info in category_replacements.items():
                print(f"\n  {col}:")
                print(f"    Original zeros: {info['original_zeros']}")
                print(f"    Zero percentage: {info['zero_percentage']:.1f}%")
                print(f"    Replacement value: {info['replacement_value']:.3f}")
                print(f"    Base year zero: {info['base_year_zero']}")
                
                if 'original_stats' in info:
                    print(f"    Original min (non-zero): {info['original_stats']['min']:.3f}")
                    print(f"    Original median: {info['original_stats']['median']:.3f}")
                
                changes = pd.DataFrame({
                    'Year': df.loc[df[col] == 0, 'Year'],
                    'Original': 0,
                    'Cleaned': info['replacement_value']
                }).head(3)
                if not changes.empty:
                    print("\n    Sample changes:")
                    print(changes)
        else:
            print("  No zero replacements needed")

def main():
    sectors = {
        'Business Enterprises': pd.read_csv('business_enterprises_data_1998_2017.csv'),
        'Government': pd.read_csv('government_data_1998_2017.csv'),
        'Higher Education': pd.read_csv('higher_education_data_1998_2017.csv'),
        'Private Non-Profit': pd.read_csv('private_non-profit_data_1998_2017.csv')
    }
    
    cleaned_sectors = {}
    for sector_name, df in sectors.items():
        print(f"\nProcessing {sector_name}:")
        
        categories = categorize_columns(df)
        
        zero_patterns = analyze_zero_patterns(df)
        
        df_cleaned, replacements = handle_zeros(df, zero_patterns, categories)
        
        validation = validate_cleaning(df, df_cleaned, replacements)
        
        print_enhanced_summary(df, df_cleaned, categories, replacements, validation)
        
        cleaned_sectors[sector_name] = {
            'data': df_cleaned,
            'replacements': replacements,
            'validation': validation
        }
        
        output_filename = f"{sector_name.lower().replace(' ', '_')}_cleaned.csv"
        df_cleaned.to_csv(output_filename, index=False)
        print(f"\nSaved cleaned data to {output_filename}")
    
    return cleaned_sectors

if __name__ == "__main__":
    cleaned_sectors = main()

