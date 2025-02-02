# Cyprus_RnD_GDP
Code and dataset for analyzing the impact of sector-specific R&amp;D investments on GDP growth in Cyprus (1998–2017) using regression modeling and feature selection techniques.

# Sector-Specific R&D Contributions to GDP Growth in Cyprus

📌 **Analyzing the impact of sectoral R&D investments on Cyprus's GDP (1998–2017) using statistical modeling, machine learning, and time-series regression techniques.**  

## 📖 Overview  
This repository contains the **dataset** and **Python scripts** used for the research project analyzing how R&D investments in different sectors (Government, Higher Education, Business Enterprises, and Non-Profit) contribute to GDP growth in Cyprus. The study applies **feature selection techniques**, **regression models**, and **time-lagged analysis** to uncover key relationships.  

---

## 📊 Dataset Description  
- **Source:** Cyprus Government Open Data Portal & World Bank  
- **Years Covered:** 1998–2017  
- **Sectors:**  
  - Government  
  - Higher Education  
  - Business Enterprises  
  - Non-Profit  
- **Key Variables:**  
  - Labour Costs  
  - Researchers  
  - Capital Expenditure  
  - Basic & Applied Research  
  - GDP Growth  

---

## ⚙️ Installation & Dependencies  
Before running the scripts, install the required libraries:  
```bash
pip install pandas numpy scikit-learn matplotlib seaborn openpyxl

python scripts/data_preprocessing.py  
python scripts/adjust_exchange_rates.py  
python scripts/prepare_gdp_data.py  
python scripts/clean_before_percentage_change.py  
python scripts/percentage_change.py  

python scripts/exploratory_data_analysis.py  

python scripts/feature_selection_outlier.py   # Outlier-based selection  
python scripts/feature_selection_hierarchical.py  # Hierarchical selection 

python scripts/regression_prequential.py         # Prequential Regression  
python scripts/regression_random_forest.py       # Random Forest Regression  
python scripts/regression_bayesian.py            # Bayesian Regression  

python scripts/regression_improvement_elastic_net.py      # Elastic Net Regularization  
python scripts/regression_improvement_ridge.py            # Ridge Regression  
python scripts/regression_improvement_time_lagged_prequential.py # Time-Lagged Prequential Regression  

 Eleftheriou, D. (2025). Sector-Specific R&D Contributions to GDP Growth in Cyprus. GitHub. Available at: https://github.com/yourusername/Cyprus-RnD-GDP
