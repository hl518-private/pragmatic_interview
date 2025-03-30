# Task.py - Air Pollution Analysis and Anomaly Detection

This script provides a comprehensive analysis of air pollution data, focusing on PM2.5 levels, as a response to Pragmatic semiconductors interview preparation task.
It includes data preparation, correlation analysis, regression modeling, and anomaly detection. The script is designed to process datasets from the WHO and OECD, providing insights into global and regional pollution trends.

## Features

### Utility Functions (`utils` Class)
1. **Spearman Correlation**:
   - Calculates the Spearman correlation matrix for a given dataset.
   - Converts non-numeric columns to categorical codes for analysis.

2. **Random Forest Regressor**:
   - Performs Random Forest regression to determine feature importance.
   - Splits data into training and testing sets and returns sorted feature importance.

3. **Flag Anomalies**:
   - Flags anomalies in the dataset based on the correlation matrix.
   - Identifies mismatches between PM2.5 levels and correlated factors.

4. **Prepare Data**:
   - Merges and cleans datasets from WHO and OECD.
   - Returns a pivoted DataFrame and a list of target variables for analysis.

5. **Plot Results**:
   - Visualizes results as bar charts with customizable titles and labels.

### Task Execution (`TaskController` Class)
1. **Global and Regional Trends**:
   - Analyzes and visualizes PM2.5 levels across countries, regions, urbanization levels, and GDP per capita.

2. **Spearman Correlation**:
   - Performs Spearman correlation analysis on the dataset.
   - Visualizes the top and bottom correlations for target variables.

3. **Random Forest Regressor**:
   - Uses Random Forest regression to identify the most important features influencing PM2.5 levels.
   - Visualizes the top and bottom feature importances.

4. **Detect Anomalies**:
   - Detects anomalies in the dataset based on correlation values and PM2.5 level categories.

5. **Main Function**:
   - Executes the above tasks in sequence:
     1. Prepares the dataset.
     2. Analyzes global and regional trends.
     3. Performs Spearman correlation analysis.
     4. Runs Random Forest regression.
     5. Detects anomalies.

## Data Sources
- **WHO Air Pollution Data**: Concentrations of fine particulate matter (PM2.5) per country.
- **OECD Country Statistics**: Economic, demographic, and environmental indicators.

## Usage
1. Place the required datasets (`data.csv` and `OECD,DF_FACTBOOK2015_PUB,+all.csv`) in the same directory as the script. You can either directly get these from the data source or unzip the uploaded zip file.
2. Run the script:
   ```bash
   python task.py
   ```
3. Follow the outputs and visualizations for insights into PM2.5 levels and related factors.

## Dependencies
- `pandas`
- `numpy`
- `matplotlib`
- `scikit-learn`

Install dependencies using:
```bash
pip install pandas numpy matplotlib scikit-learn
```

## Outputs
- Visualizations of trends, correlations, and feature importances.
- Anomalies detected in the dataset based on correlations.

## Notes
- Ensure the datasets are formatted correctly and contain the required columns for merging and analysis.
- The script saves intermediate cleaned and pivoted datasets as CSV files for reference.

## Author
Hanna Lee
This script was developed to analyze air pollution data and detect anomalies using statistical and machine learning techniques.
