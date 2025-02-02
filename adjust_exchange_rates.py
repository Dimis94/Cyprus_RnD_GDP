#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import os

# Define constants
INPUT_FILE = 'API_PA.NUS.FCRF_DS2_en_csv_v2_3403488.csv'
OUTPUT_FILE = 'cyprus_exchange_rate_1998_2017.csv'
START_YEAR = 1998
END_YEAR = 2017

def process_exchange_rates(input_file, output_file, start_year, end_year):
    # Check if input file exists
    if not os.path.exists(input_file):
        raise FileNotFoundError(f"Input file {input_file} not found.")

    # Load the exchange rate data file
    exchange_rate_df = pd.read_csv(input_file, skiprows=4)

    # Filter for Cyprus exchange rate data
    cyprus_exchange_rate = exchange_rate_df[exchange_rate_df['Country Name'] == 'Cyprus']

    # Verify that Cyprus data is present
    if cyprus_exchange_rate.empty:
        raise ValueError("No data found for Cyprus in the input file.")

    # Select the columns for the specified year range
    year_columns = [str(year) for year in range(start_year, end_year + 1)]
    selected_columns = ['Country Name', 'Country Code'] + year_columns
    cyprus_exchange_rate_selected = cyprus_exchange_rate[selected_columns]

    # Reshape the data so that years are in rows (transpose)
    cyprus_exchange_rate_transposed = cyprus_exchange_rate_selected.melt(
        id_vars=['Country Name', 'Country Code'],
        var_name='Year', 
        value_name='Exchange Rate (LCU per USD)'
    )

    # Save the reshaped exchange rate data to a CSV file
    cyprus_exchange_rate_transposed.to_csv(output_file, index=False)
    print(f"Exchange rates for Cyprus ({start_year}-{end_year}) saved to {output_file}")

# Execute the function
try:
    process_exchange_rates(INPUT_FILE, OUTPUT_FILE, START_YEAR, END_YEAR)
except Exception as e:
    print(f"An error occurred: {str(e)}")

