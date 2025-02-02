#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import os

# Define constants
GDP_INPUT_FILE = 'API_NY.GDP.MKTP.CD_DS2_en_csv_v2_3403845.csv'
EXCHANGE_RATE_FILE = 'cyprus_exchange_rate_1998_2017.csv'
OUTPUT_FILE = 'cyprus_gdp_1998_2017_lcu.csv'
START_YEAR = 1998
END_YEAR = 2017

def process_gdp_data(gdp_file, exchange_rate_file, output_file, start_year, end_year):
    # Check if input files exist
    for file in [gdp_file, exchange_rate_file]:
        if not os.path.exists(file):
            raise FileNotFoundError(f"Input file {file} not found.")

    # Step 1: Load the GDP data file
    gdp_df = pd.read_csv(gdp_file, skiprows=4)

    # Filter for Cyprus GDP data
    cyprus_gdp = gdp_df[gdp_df['Country Name'] == 'Cyprus']

    # Verify that Cyprus data is present
    if cyprus_gdp.empty:
        raise ValueError("No GDP data found for Cyprus in the input file.")

    # Select the relevant columns: Years from start_year to end_year and GDP in USD
    year_columns = [str(year) for year in range(start_year, end_year + 1)]
    selected_columns = ['Country Name', 'Country Code'] + year_columns
    cyprus_gdp_selected = cyprus_gdp[selected_columns]

    # Reshape the GDP data so that years are in rows
    cyprus_gdp_transposed = cyprus_gdp_selected.melt(
        id_vars=['Country Name', 'Country Code'],
        var_name='Year', 
        value_name='GDP (USD)'
    )

    # Ensure the 'Year' column is treated as a string in the GDP data
    cyprus_gdp_transposed['Year'] = cyprus_gdp_transposed['Year'].astype(str)

    # Step 2: Load the exchange rate data
    exchange_rate_df = pd.read_csv(exchange_rate_file)

    # Ensure the 'Year' column is treated as a string in the exchange rate data
    exchange_rate_df['Year'] = exchange_rate_df['Year'].astype(str)

    # Step 3: Merge the GDP data with the exchange rate data on 'Year'
    merged_data = pd.merge(cyprus_gdp_transposed, exchange_rate_df, on=['Country Name', 'Country Code', 'Year'])
   
    # Step 4: Convert GDP from USD to Local Currency (LCU) or EUR
    def convert_currency(row):
        if int(row['Year']) < 2008:
            return row['GDP (USD)'] * row['Exchange Rate (LCU per USD)']
        else:
            return row['GDP (USD)'] * row['Exchange Rate (LCU per USD)']  # Now represents EUR per USD

    merged_data['GDP (LCU/EUR)'] = merged_data.apply(convert_currency, axis=1)

    # Update column names
    merged_data['Currency'] = merged_data['Year'].apply(lambda x: 'LCU' if int(x) < 2008 else 'EUR')

    # Step 5: Save the final data to a CSV file
    columns_to_save = ['Country Name', 'Country Code', 'Year', 'GDP (USD)', 
                       'Exchange Rate (LCU per USD)', 'GDP (LCU/EUR)', 'Currency']
    merged_data[columns_to_save].to_csv(output_file, index=False)
    print(f"GDP data for Cyprus ({start_year}-{end_year}) converted to LCU/EUR and saved to {output_file}")

# Execute the function
try:
    process_gdp_data(GDP_INPUT_FILE, EXCHANGE_RATE_FILE, OUTPUT_FILE, START_YEAR, END_YEAR)
except Exception as e:
    print(f"An error occurred: {str(e)}")

