import pandas as pd
from enum import Enum
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np


def spearman_correlation(df):
    """
    Calculate the Spearman correlation and return the correlation matrix.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data to be processed.

    Returns:
    correlation_matrix_spearman (pd.DataFrame): Resulting DataFrame with the Spearman correlation matrix.
    """
    # Convert non-numeric columns to categorical codes
    non_numeric_data = df.select_dtypes(exclude=['number']).columns
    for col in non_numeric_data:
        df[col] = df[col].astype('category').cat.codes

    # Calculate correlation matrix using pairwise deletion, i.e. ignoring NaN values
    correlation_matrix_spearman = df.corr(method='spearman', min_periods=1)
    
    return correlation_matrix_spearman

def random_forest_regressor(X, y):
    """
    Perform Random Forest regression and return the sorted feature importance.

    Parameters:
    X (pd.DataFrame): Features for regression.
    y (pd.Series): Target variable for regression.

    Returns:
    sorted_importance (pd.Series): Sorted feature importance from the Random Forest model.
    """
    # Split data into training and testing sets
    # Note: Random forest regressor doesn't require scaling of data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train) 

    # Get feature importance
    feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
    sorted_importance = feature_importances.sort_values(ascending=False)

    return sorted_importance

def flag_anomalies(df, correlation_matrix):
    """
    Flag anomalies in the dataset based on correlation matrix.

    Parameters:
    df (pd.DataFrame): The DataFrame containing data to be processed.
    correlation_matrix (pd.DataFrame): OECD indicators that most impacts mean PM2.5 level

    Returns:
    flags(list): list with anomalies flagged
    """
    flags = []
    low_flag = ['very low', 'low', 'medium']
    high_flag = ['high', 'very high', 'medium']
    for factor in correlation_matrix.index:
        correlation = correlation_matrix[factor]
        for index, row in df.iterrows():
            dust_level = row['Mean PM2.5 Level']
            if (correlation > 0 and dust_level in low_flag) or (correlation < 0 and dust_level in high_flag):
                flags.append((row['Country'], row['Year'], factor, dust_level, correlation))
    
    return flags

def prepare_data():
    """
    The function merges the two datasets, cleans up the data, and pivots it for analysis.
        data.csv (SOURCE: https://www.who.int/data/gho/data/indicators/indicator-details/GHO/concentrations-of-fine-particulate-matter-(pm2-5))
        OECD,DF_FACTBOOK2015_PUB,+all.csv (SOURCE: https://data-explorer.oecd.org/vis?tenant=archive&df[ds]=DisseminateArchiveDMZ&df[id]=DF_FACTBOOK2015_PUB&df[ag]=OECD&dq=.&pd=2010%2C2015&to[TIME_PERIOD]=false&vw=ov)
    The function returns the cleaned and pivoted DataFrame along with a list of target variables.
    
    Parameters:
    None

    Returns:
    pivoted_df (pd.DataFrame): Resulting DataFrame with the cleaned and pivoted data.
    target_variables (list): List of target variables for analysis.
    """
    # Load and merge dataset
    df1 = pd.read_csv("data.csv", low_memory=False)                            # Load WHO Air Polution Data - contains concentrations of fine particulate matter per country
    df2 = pd.read_csv("OECD,DF_FACTBOOK2015_PUB,+all.csv", low_memory=False)   # Load OECD Country Statistics - contains economic, demographic, and environmental indicators
    df1 = df1.rename(columns={'Location': 'Country', 'Period': 'TIME_PERIOD'})                          # Match column names
    merged_df = df1.merge(df2, on=['Country', 'TIME_PERIOD'], how='left').dropna(subset=['OBS_VALUE'])  # Merge based on country and year and drop rows with no OECD results
    # merged_df.to_csv("merged_data.csv", index=False)

    # Cleanup with relevant columns, give more meaningful column names, and add column for PM2.5 level range
    # Note: the WHO website doesn't specify what FactValueNumericLow and FactValueNumericHigh are. Assuming that they are the lower and upper bounds of the PM2.5 level
    clean_df = merged_df[['ParentLocation', 'Country', 'Dim1', 'TIME_PERIOD', 'Subject', 'OBS_VALUE', 'FactValueNumeric', 'FactValueNumericLow', 'FactValueNumericHigh']].dropna()
    clean_df = clean_df.rename(columns={'ParentLocation': 'Region', 
                                        'Dim1': 'Urbanization Level',
                                        'TIME_PERIOD': 'Year',
                                        'Subject': 'OECD indicator',
                                        'OBS_VALUE': 'Observed Value',
                                        'FactValueNumeric': 'Mean PM2.5 Level',
                                        'FactValueNumericLow': 'Minimum PM2.5 Level',
                                        'FactValueNumericHigh': 'Maximum PM2.5 Level'})
    clean_df['PM2.5 Level Range'] = clean_df['Maximum PM2.5 Level'] - clean_df['Minimum PM2.5 Level']
    # clean_df.to_csv("clean_.csv", index=False)

    # Put each 'OECD indicator' values into columns and assign corresponding 'Observed Value' to them
    index_columns = [col for col in clean_df.columns if col not in ['OECD indicator', 'Observed Value']]
    pivoted_df = clean_df.pivot_table(index=index_columns, columns='OECD indicator', values='Observed Value', aggfunc='first')
    pivoted_df = pivoted_df.reset_index()
    pivoted_df.to_csv("pivoted_data.csv", index=False)

    target_variables = ['Mean PM2.5 Level', 'Minimum PM2.5 Level', 'Maximum PM2.5 Level', 'PM2.5 Level Range']
    
    return pivoted_df, target_variables

def plot_res(res_matrix, title, xlabel):
    """
    Plot the results as bar chart.

    Parameters:
    res_matrix (pd.Series): The result matrix to be plotted.
    title (str): The title of the plot.
    xlabel (str): The label for the x-axis.

    Returns:
    None: Displays the plot.
    """
    plt.figure(figsize=(20, 8)) 
    colors = plt.cm.Blues(np.linspace(0.9, 0.2, len(res_matrix)))
    res_matrix.plot(kind='barh', color=colors, alpha=0.7)
    
    # Split y-tick labels into multiple lines
    max_length = 50  # Maximum number of characters per line
    new_labels = []
    for label in res_matrix.index:
        words = label.split()
        lines = []
        current_line = ""
        for word in words:
            if len(current_line + " " + word) <= max_length:
                current_line += " " + word if current_line else word
            else:
                lines.append(current_line)
                current_line = word
        if current_line:
            lines.append(current_line)
        new_labels.append("\n".join(lines))  # Join the lines with newline characters
    plt.yticks(ticks=np.arange(len(res_matrix.index)), labels=new_labels)

    plt.title(title)
    plt.ylabel(res_matrix.index.name, labelpad=0.5)
    plt.xlabel(xlabel)
    plt.tight_layout()
    plt.subplots_adjust(left=0.4, right=0.95)
    plt.show()

def main():
    print("Preparing dataset...")
    prepared_df, target_variables = prepare_data()
    non_numeric_data = prepared_df.select_dtypes(exclude=['number']).columns
    print("Dataset prepared.")

    breakpoint()

    # STEP 1: GLOBAL AND REGIONAL TRENDS
    # Pollution levels across countries, regions, urbanization level and income groups (by GDP per capita)
    print("####################################################################  GLOBAL AND REGIONAL TRENDS  ####################################################################")
    plotting_df = prepared_df.copy()

    # Plotting the mean PM2.5 level by country
    mean_pm25_by_country = plotting_df.groupby('Country')['Mean PM2.5 Level'].mean().sort_values(ascending=False)
    top_10 = mean_pm25_by_country.head(10)
    bottom_10 = mean_pm25_by_country.tail(10)
    top_bottom_10 = pd.concat([top_10, bottom_10])
    plot_res(top_bottom_10, title='Top 10 and Bottom 10 Countries by Mean PM2.5 Level', xlabel='Mean PM2.5 Level')
    plt.show()
 
    # Plotting the mean PM2.5 level by region
    region_means = plotting_df.groupby('Region')['Mean PM2.5 Level'].mean().sort_values()
    plt.bar(region_means.index, region_means.values, color='green', alpha=0.7)
    plt.xlabel('Region')
    plt.ylabel('Mean PM2.5 Level')
    plt.title('Mean PM2.5 Level by Region')
    plt.show()

    # Plotting the mean PM2.5 level by urbanization level
    urbanization_means = plotting_df.groupby('Urbanization Level')['Mean PM2.5 Level'].mean().sort_values()
    plt.bar(urbanization_means.index, urbanization_means.values, color='purple', alpha=0.7)
    plt.xlabel('Urbanization Level')
    plt.ylabel('Mean PM2.5 Level')
    plt.title('Mean PM2.5 Level by Urbanization Level')
    plt.show()

    # Plotting the mean PM2.5 level by GDP per capita
    gdp_means = plotting_df.groupby('GDP per capita')['Mean PM2.5 Level'].mean().sort_values()
    plt.plot(gdp_means.index, gdp_means.values)
    plt.xlabel('GDP per capita')
    plt.ylabel('Mean PM2.5 Level')
    plt.title('Mean PM2.5 Level by GDP per capita')
    plt.show()

    breakpoint()
    # STEP 2: SPEARMAN CORRELATION
    # Simpletest method to get correlation value from -1 to +1 are Spearman or Pearson.
    # Pearson is used for linear correlation, while spearman is used for non-linear, rank-based correlation.
    print("####################################################################  SPEARMAN CORRELATION  ####################################################################")
    df_spearman = prepared_df.copy()
    corr_matrix = spearman_correlation(df_spearman)
    xlabel = "Correlation Value"
    got_data_for_anomalie_detection = False
    for target in target_variables:
        # Select interested columns, then sort the selected columns by descending correlation values
        selected_columns = corr_matrix[target]
        sorted_selected_columns = selected_columns.loc[selected_columns.abs().sort_values(ascending=False).index]
        sorted_selected_columns = sorted_selected_columns.drop(index=target_variables)
        if not got_data_for_anomalie_detection:
            correlation_matrix_for_anomalie_detection = sorted_selected_columns.copy()
        print(f"##################################  Sorted correlations for selected column: {target}  ##################################")
        top_10 = sorted_selected_columns.head(10)
        print("Top 10\n", top_10, "\n")
        plot_res(top_10, f"Top 10 correlations to {target}", xlabel)
        bottom_10 = sorted_selected_columns.tail(10)
        bottom_10_reversed = bottom_10.iloc[::-1]
        print("Bottom 10\n", bottom_10, "\n")
        plot_res(bottom_10, f"Bottom 10 correlations to {target}", xlabel)

        # Check what correlation values we're getting for non-numeric data
        encoded_correlation = sorted_selected_columns.loc[non_numeric_data.tolist()]
        print("Correlations for non-numeric data:\n", encoded_correlation)
        
    breakpoint()
    # STEP 3: RANDOM FOREST REGRESSOR
    # Try forest regressor model on non-numeric data separately which can give importance factors for multiple features.
    print("####################################################################  RANDOM FOREST REGRESSOR  ####################################################################")
    df_RFR = prepared_df.copy()
    columns_to_keep = non_numeric_data.tolist() + target_variables
    df_RFR = df_RFR[columns_to_keep]
    df_RFR = pd.get_dummies(df_RFR, columns=non_numeric_data, drop_first=True)  # One-hot encode non-numeric variables
    xlabel = "Locations"
    X = df_RFR.drop(columns=target_variables)  # Features (excluding all target variables)
    for target in target_variables:
        y = df_RFR[target]
        sorted_importance = random_forest_regressor(X, y)
        print(f"##################################  Feature Importance for {target}  ##################################")
        top_10 = sorted_importance.head(10)
        print("Top 10 Features:\n", top_10, "\n")
        plot_res(top_10, f"Top 10 importance to {target}", xlabel)

        bottom_10 = sorted_importance.tail(10)
        bottom_10_reversed = bottom_10.iloc[::-1]
        print("Bottom 10 Features\n", bottom_10, "\n")
        plot_res(bottom_10_reversed, f"Bottom 10 importance to {target}", xlabel)

    breakpoint()
    # STEP 4: DETECT ANOMALIES
    # Detect anomalies based on correlations that impacts mean PM2.5 level
    print("####################################################################  DETECT ANOMALIES  ####################################################################")
    print("Finding anomalies...")
    bins = [0, 10, 15, 25, 35, 100]
    labels = ['very low', 'low', 'medium', 'high', 'very high']
    prepared_df['categorised PM2.5 level'] = pd.cut(prepared_df['Mean PM2.5 Level'], bins=bins, labels=labels)
    
    flags = flag_anomalies(prepared_df, correlation_matrix_for_anomalie_detection)
    if flags == []:
        print("No anomalies detected.")
    else:
        print("Anomalies detected:")
        for flag in flags:
            print(f"Country: {flag[0]}, Year: {flag[1]}, OECD Indicator: {flag[2]}, Dust Level: {flag[3]}, Correlation: {flag[4]}")
        

if __name__ == "__main__":
    main()