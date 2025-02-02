#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

gov_df = pd.read_csv('government_data_1998_2017.csv')
bus_df = pd.read_csv('business_enterprises_data_1998_2017.csv')
edu_df = pd.read_csv('higher_education_data_1998_2017.csv')
pnp_df = pd.read_csv('private_non-profit_data_1998_2017.csv')
gdp_df = pd.read_csv('cyprus_gdp_1998_2017_lcu.csv')

# Function to normalize data
def normalize_data(df, base_year=1998):
    normalized_df = df.copy()
    for column in df.select_dtypes(include=[np.number]).columns:
        if column not in ['Year']:
            base_value = df[df['Year'] == base_year][column].values[0]
            if base_value != 0:  
                normalized_df[column] = (df[column] / base_value - 1) * 100
            else:
                normalized_df[column] = 0  
    return normalized_df

gov_norm = normalize_data(gov_df)
bus_norm = normalize_data(bus_df)
edu_norm = normalize_data(edu_df)
pnp_norm = normalize_data(pnp_df)
gdp_norm = normalize_data(gdp_df)

# 1. Time series plot of total researchers across sectors (normalized)
plt.figure(figsize=(12, 6))
for df, name in zip([gov_norm, bus_norm, edu_norm, pnp_norm], ['Government', 'Business', 'Education', 'Non-Profit']):
    plt.plot(df['Year'], df['Researchers'], label=name)
plt.title('Normalized Growth in Total Researchers by Sector (1998-2017)')
plt.xlabel('Year')
plt.ylabel('Percentage Change from 1998')
plt.legend()
plt.grid(True)
plt.show()

# 2. R&D Expenditure Growth vs GDP Growth
plt.figure(figsize=(12, 6))
for df, name in zip([gov_norm, bus_norm, edu_norm, pnp_norm], ['Government', 'Business', 'Education', 'Non-Profit']):
    total_expenditure = df['Labour Costs'] + df['Capital Expenditure'] + df['Other Current Expenditure']
    plt.plot(df['Year'], total_expenditure, label=name)
plt.plot(gdp_norm['Year'], gdp_norm['GDP (LCU/EUR)'], label='GDP', linewidth=3, color='black')
plt.title('Normalized R&D Expenditure Growth vs GDP Growth (1998-2017)')
plt.xlabel('Year')
plt.ylabel('Percentage Change from 1998')
plt.legend()
plt.grid(True)
plt.show()

# 3. Heatmap of correlations between key variables (using Government sector as an example)
key_vars = ['Researchers', 'PhD Holders', 'Labour Costs', 'Capital Expenditure', 
            'Basic Research', 'Applied Research', 'Experimental Development']
correlation_matrix = gov_norm[key_vars].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0)
plt.title('Correlation Heatmap of Key Variables - Government Sector')
plt.tight_layout()
plt.show()

# 4. Box plot of R&D intensity across sectors
def calculate_rd_intensity(df, gdp_df):
    total_expenditure = df['Labour Costs'] + df['Capital Expenditure'] + df['Other Current Expenditure']
    return (total_expenditure / gdp_df['GDP (LCU/EUR)']) * 100

sectors = ['Government', 'Business', 'Education', 'Non-Profit']
intensities = [calculate_rd_intensity(df, gdp_df) for df in [gov_df, bus_df, edu_df, pnp_df]]

plt.figure(figsize=(10, 6))
plt.boxplot(intensities, labels=sectors)
plt.title('R&D Intensity Distribution by Sector (1998-2017)')
plt.ylabel('R&D Intensity (% of GDP)')
plt.grid(True)
plt.show()

# 5. Stacked area chart of normalized funding sources for Government sector
funding_columns = ['Government Budget', 'Self Financing', 'Business Enterprises', 
                   'Research and Innovation Foundation', 'European Union', 'Other Sources from Abroad']
funding_data = gov_norm[['Year'] + funding_columns].set_index('Year')

# Stacked area chart of normalized funding sources for Government sector without stacking
plt.figure(figsize=(12, 6))
funding_data.plot.area(stacked=False)  # Stacked is set to False
plt.title('Normalized Funding Sources - Government Sector (1998-2017)')
plt.xlabel('Year')
plt.ylabel('Percentage Change from 1998')
plt.legend(title='Funding Source', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()


# Print summary statistics
print("\nSummary Statistics for Normalized Data (Percentage Change from 1998):")
for name, df in zip(['Government', 'Business', 'Education', 'Non-Profit'], 
                    [gov_norm, bus_norm, edu_norm, pnp_norm]):
    print(f"\n{name} Sector:")
    print(df[['Researchers', 'Labour Costs', 'Capital Expenditure', 'Other Current Expenditure']].describe())

