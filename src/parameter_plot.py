#PLOT ANALYZE 2

import pandas as pd
import os
import requests
import geopandas as gpd
import time
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
DEPARTEMENT_CODE = '17'
DATA_DIR = 'data'
PLOTS_DIR = 'plots'

def load_data_list(file_path='data/products/file_row_counts.csv', debug=False):
    if debug:
        print(f"Load and prepare data from {file_path}")
    df = pd.read_csv(file_path)
    if debug:
        print("First 5 rows of the DataFrame:")
        print(df.head())
        print("\nNumber of rows in the original DataFrame:", len(df))
    df = df[(df['mean'] != 0.0) & (df['zero_percentage'] < 80.0) & (df['rows'] > 10000) & (df['unique_val'] > 10)]
    if debug:
        print("Number of rows in the filtered DataFrame:", len(df))
    return df['file_name'].to_list()

def load_and_prepare_data(file_name, debug=False):
    df = pd.read_csv(file_name)
    df['dateprel'] = pd.to_datetime(df['dateprel'])
    #cddept,inseecommuneprinc,nomcommuneprinc,dateprel,heureprel,cdparametre,cdunitereferencesiseeaux,limitequal,valtraduite,NomParametre,LbCourtParametre
    #str, str, str, datetime64[ns], str, str, str, str, float, str, str
    df['inseecommuneprinc'] = df['inseecommuneprinc'].astype(str)
    df['valtraduite'] = pd.to_numeric(df['valtraduite'], errors='coerce')
    df = df.dropna(subset=['valtraduite'])
    df['cdparametre'] = df['cdparametre'].astype(str)
    df['cdunitereferencesiseeaux'] = df['cdunitereferencesiseeaux'].astype(str)
    df['cddept'] = df['cddept'].astype(str)
    df['LbCourtParametre'] = df['LbCourtParametre'].astype(str)
    return df

def fetch_geodata(departement_code=None):
    if departement_code is None:
        return None
    """Fetches GeoJSON data for a given departement code with retry logic."""
    url = GEO_API_URL.format(departement_code)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
            data = response.json()
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            gdf.rename(columns={'code': 'inseecommuneprinc'}, inplace=True)
            gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
            return gdf
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                delay = BACKOFF_FACTOR ** attempt
                time.sleep(delay)
            else:
                print("Max retries reached. Failed to fetch geodata.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return None

def map(df, departement, gdf=None, cdparametre=None, debug=False):
    if cdparametre is not None:
        df = df[df['cdparametre'] == cdparametre]
    df = df[df['cddept'] == departement]
    if len(df) == 0:
        print(f"No data available for departement {departement} and parameter {cdparametre}. Exiting.")
        return
    if gdf is None:
        gdf = fetch_geodata(departement_code=departement)
    if gdf is None:
        print("Failed to fetch geodata. Exiting.")
        return
    
    df_average = df.groupby('inseecommuneprinc')['valtraduite'].mean().reset_index(name='average_valtraduite')
    df_min = df_average['average_valtraduite'].min()
    df_max = df_average['average_valtraduite'].max()
    
    merged = gdf.merge(df_average, on='inseecommuneprinc', how='left')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    
    # Use a separate object for the plot to configure the colorbar
    plot = merged.plot(
        column='average_valtraduite',
        cmap='viridis',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': f"Value ({df['cdunitereferencesiseeaux'].iloc[0]})", 'orientation': "horizontal"},
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
        vmin=df_min,  # Set the minimum value for the color scale
        vmax=df_max   # Set the maximum value for the color scale
    )
    
    ax.set_title(f"{df['LbCourtParametre'].iloc[0]}", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    path = os.path.join(PLOTS_DIR, f'{df['LbCourtParametre'].iloc[0]}')
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    plt.savefig(f'{path}/map.png')
    plt.close()

def plot_yearly_monthly_average(df, year_col='valtraduite', month_col='centered_reduced_val', save=False):
    # Ensure 'dateprel' is datetime
    df['dateprel'] = pd.to_datetime(df['dateprel'])

    # Extract year and month
    df['year'] = df['dateprel'].dt.year
    df['month'] = df['dateprel'].dt.month

    # Calculate the yearly and monthly averages
    yearly_averages = df.groupby('year')[year_col].mean()
    monthly_averages = df.groupby('month')[month_col].mean()

    # Calculate additional information
    total_entries = len(df)
    #avg_monthly_entries = df.groupby('month').size().mean()
    #avg_yearly_entries = df.groupby('year').size().mean()
    monthly_entries = df.groupby('month').size()
    yearly_entries = df.groupby('year').size()

    # Create subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    # Plot yearly averages as a line plot
    years_range = range(2019, 2025)
    ax1.plot(yearly_averages.index, yearly_averages.values, marker='o', linestyle='-', color='blue')
    ax1.set_title(f'Yearly Averages of {df["LbCourtParametre"].iloc[0]} \n'
                  f'Total Entries: {total_entries}')
    ax1.set_ylabel(f'Average Value ({df["cdunitereferencesiseeaux"].iloc[0]})')
    ax1.set_xlabel('Year')
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xticks(years_range)
    ax1.tick_params(axis='x', rotation=0)
    ax1.set_xlim(min(years_range), max(years_range))
    # Annotate each year with the number of entries
    for year, count in yearly_entries.items():
        ax1.annotate(f'Entries: {count}', xy=(year, yearly_averages[year]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    # Plot monthly averages as a bar plot
    colors = ['green' if x >= 0 else 'red' for x in monthly_averages]  # Green for positive, red for negative
    ax2.bar(monthly_averages.index, monthly_averages.values, color=colors)
    ax2.set_title(f'Monthly Averages of {df["LbCourtParametre"].iloc[0]}\n'
                  f'Total Entries: {total_entries}')
    ax2.set_ylabel('Average Centered Reduced Value (No Unit)')
    ax2.set_xlabel('Month')
    ax2.set_xticks(monthly_averages.index)
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(axis='y', linestyle='--')
    # Annotate each month with the number of entries
    for month, count in monthly_entries.items():
        ax2.annotate(f'Entries: {count}', xy=(month, monthly_averages[month]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()  # Adjust layout to prevent labels from overlapping

    # Save the plot
    if save:
        path = os.path.join(PLOTS_DIR, f'{df["LbCourtParametre"].iloc[0]}')
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(f'{path}/Yearly_Monthly_Averages.png')
    else:
        plt.show()
    plt.close()

    def cross_correlation(series1, series2, max_lag, debug=False):
        """
        Calculates the cross-correlation between two time series with increasing lag.

        Args:
            series1 (pd.Series): The first time series.
            series2 (pd.Series): The second time series.
            max_lag (int): The maximum lag to consider.

        Returns:
            pd.DataFrame: A DataFrame containing the lag and correlation values.
        """
        if debug:
            print(f"Calculating cross-correlation with max_lag={max_lag}")
        correlations = []
        lags = range(-max_lag, max_lag + 1)
        for lag in lags:
            if lag < 0:
                shifted_series1 = series1[-lag:]
                shifted_series2 = series2[:lag]
            elif lag > 0:
                shifted_series1 = series1[:-lag]
                shifted_series2 = series2[lag:]
            else:
                shifted_series1 = series1
                shifted_series2 = series2
            
            min_length = min(len(shifted_series1), len(shifted_series2))
            
            if min_length == 0:
                corr = np.nan
            else:
                corr = np.corrcoef(shifted_series1[-min_length:], shifted_series2[-min_length:])[0, 1]
            
            correlations.append(corr)
        
        return pd.DataFrame({'lag': lags, 'correlation': correlations})

def plot_time_series(df, col='valtraduite', save=False, debug=False):
    if debug:
        print(f"Plotting time series for {col}")
    mean = df[col].mean()
    std = df[col].std()
    if debug:
        print(f"Mean: {mean}, Std: {std}")
    df = df.sort_values(by='dateprel')
    ax = df.plot(x='dateprel', y=col, figsize=(12, 6), title=f'Time Series of {col}')
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.axhline(mean, color='r', linestyle='--', label=f'Mean = {mean:.2f}')
    plt.axhline(mean + std, color='g', linestyle='--', label=f'Mean + Std = {mean + std:.2f}')
    plt.axhline(mean - std, color='g', linestyle='--', label=f'Mean - Std = {mean - std:.2f}')
    plt.legend()
    if save:
        if debug:
            print(f"Saving plot to {path}")
        path = os.path.join(PLOTS_DIR, f'{df['LbCourtParametre'].iloc[0]}')
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(f'{path}/centered_reduced.png')
        plt.close()
    else:
        if debug:
            print("Showing plot")
        plt.show()
    plt.close()

def smooth(df, window_size=60, col='centered_reduced_val'):
    df = df.set_index('dateprel').sort_index()
    df[col] = df[col].rolling(f'{window_size}D', center=True).mean()
    return df.reset_index()

def remove_outliers(df, col='valtraduite'):
    mean = df[col].mean()
    std = df[col].std()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def center_reduce(df):
    def process_commune(commune_df):
        if len(commune_df) < 50:
            return None
        filtered_df = remove_outliers(commune_df, col='valtraduite')
        mean = filtered_df['valtraduite'].mean()
        std = filtered_df['valtraduite'].std()
        centered_reduced_df = filtered_df.copy()
        centered_reduced_df['centered_reduced_val'] = (centered_reduced_df['valtraduite'] - mean) / std
        return centered_reduced_df

    communes = df['inseecommuneprinc'].unique()
    centered_reduced_dfs = [process_commune(df[df['inseecommuneprinc'] == commune]) for commune in communes]
    centered_reduced_dfs = [df for df in centered_reduced_dfs if df is not None]

    if not centered_reduced_dfs:
        return pd.DataFrame()
    
    return pd.concat(centered_reduced_dfs, ignore_index=True)