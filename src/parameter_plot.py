import pandas as pd
import requests
import geopandas as gpd
import matplotlib.pyplot as plt
import time
import os
from sklearn.linear_model import LinearRegression
from tqdm import tqdm
import numpy as np
import itertools
import seaborn as sns
from processing_functions import *
from scipy.signal import find_peaks

# Constants
DATA_DIR = 'data'
PLOTS_DIR = 'plots'
GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
DEPARTEMENT_CODE = '17'

# --- Data Loading and Filtering ---


# --- GeoData Fetching ---

def fetch_geodata(departement_code=DEPARTEMENT_CODE):
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

# --- Plotting ---

def map_parametre(gdf, average_values, min_val, max_val, parametre_info):
    """Plots the average 'valtraduite' on a map."""
    merged = gdf.merge(average_values, on='inseecommuneprinc', how='left')

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        column='average_valtraduite',
        cmap='viridis',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': f"Average {parametre_info['LbCourtParametre']} (parametre_info['cdunitereferencesiseeaux'])", 'orientation': "horizontal"},
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
        vmin=min_val,
        vmax=max_val
    )

    ax.set_title(f'{parametre_info['LbCourtParametre']} ({parametre_info['cdunitereferencesiseeaux']})', fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    path = os.path.join(PLOTS_DIR, f'{parametre_info['LbCourtParametre']}')
    os.makedirs(path, exist_ok=True)  # Ensure the directory exists
    plt.savefig(f'{path}/map.png')
    plt.close()

def map_all_parametre(df):        
    # Grouping outside the loop for efficiency
    parametre_info_df = df.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()

    average_values = df.groupby(['cdparametre', 'inseecommuneprinc'])['valtraduite'].mean().reset_index(name='average_valtraduite')
    min_max_values = df.groupby('cdparametre')['valtraduite'].agg(['min', 'max']).reset_index().rename(columns={'min': 'min_valtraduite', 'max': 'max_valtraduite'})
    geodata = fetch_geodata()

    if geodata is None:
        print("Failed to fetch geodata. Cannot generate plots.")
        return
    
    geodata['inseecommuneprinc'] = geodata['inseecommuneprinc'].astype('int64')

    for cdparametre in tqdm(df['cdparametre'].unique(), desc="Processing parameters"):
        parametre_info = parametre_info_df[parametre_info_df['cdparametre'] == cdparametre].iloc[0]
        # Filtering inside the loop
        average_values_filtered = average_values[average_values['cdparametre'] == cdparametre]
        min_max_values_filtered = min_max_values[min_max_values['cdparametre'] == cdparametre]
        
        if min_max_values_filtered.empty:
            print(f"No min/max values found for cdparametre {cdparametre}. Skipping.")
            continue
        
        min_val = min_max_values_filtered['min_valtraduite'].iloc[0]
        max_val = min_max_values_filtered['max_valtraduite'].iloc[0]
        
        map_parametre(geodata, average_values_filtered, min_val, max_val, parametre_info=parametre_info)

# --- Seasonal Analysis ---

def plot_time_series(df, valcol, mean=None, std=None, outlier_df=None, parametre_info=None, nom_plot='time_series', save=False):
    # Convert the date column to datetime if not already
    df.loc[:, 'dateprel'] = pd.to_datetime(df['dateprel'])
    df = df.sort_values('dateprel')
    
    plt.figure(figsize=(10, 6))
    plt.plot(df['dateprel'], df[valcol], marker='o', linestyle='-', label=valcol)
    
    # Plot outliers as red scatter points if provided
    if outlier_df is not None and not outlier_df.empty:
        outlier_df.loc[:, 'dateprel'] = pd.to_datetime(outlier_df['dateprel'])
        plt.scatter(outlier_df['dateprel'], outlier_df[valcol], color='red', zorder=5, label='Outliers')
    
    plt.xlabel('Date')
    plt.ylabel(valcol)
    
    # Use parameter info in the title if available
    title = f'Time Series of {parametre_info["LbCourtParametre"]} ({parametre_info["cdunitereferencesiseeaux"]})'
    plt.title(title)
    
    # Optionally plot the mean and standard deviation lines if provided
    if mean is not None and std is not None:
        if mean != 0 and std != 1:
            plt.axhline(y=mean, color='r', linestyle='--', label=f'Mean: {mean:.2f}')
            plt.axhline(y=mean + std, color='g', linestyle=':', label=f'Mean + STD: {mean + std:.2f}')
            plt.axhline(y=mean - std, color='g', linestyle=':', label=f'Mean - STD: {mean - std:.2f}')

    plt.grid(True)
    plt.tight_layout()
    plt.legend()
    
    if save:
        # Save the plot
        path = os.path.join(PLOTS_DIR, f'{parametre_info['LbCourtParametre']}')
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(f'{path}/{nom_plot}.png')
        plt.close()
    else:
        plt.show()
        plt.close()

def plot_monthly_average(df, valcol, parametre_info=None, save=False):
    # Ensure 'dateprel' is datetime
    df['dateprel'] = pd.to_datetime(df['dateprel'])

    # Extract month
    df['month'] = df['dateprel'].dt.month

    # Calculate the monthly averages
    monthly_averages = df.groupby('month')[valcol].mean()

    # Plot the monthly averages with bars, differentiating positive and negative values
    plt.figure(figsize=(12, 6))
    colors = ['green' if x >= 0 else 'red' for x in monthly_averages]  # Green for positive, red for negative
    monthly_averages.plot(kind='bar', color=colors)
    plt.title(f'Monthly Averages of {valcol} Values for {parametre_info["LbCourtParametre"]} ({parametre_info["cdunitereferencesiseeaux"]})')
    plt.ylabel(f'Average {valcol} Value')
    plt.xlabel('Month')
    plt.xticks(rotation=0)  # Keep x-axis labels horizontal
    plt.grid(axis='y', linestyle='--')  # Add a grid for better readability
    plt.tight_layout()  # Adjust layout to prevent labels from overlapping
    
    # Save the plot
    if save:
        path = os.path.join(PLOTS_DIR, f'{parametre_info['LbCourtParametre']}')
        os.makedirs(path, exist_ok=True)  # Ensure the directory exists
        plt.savefig(f'{path}/Monthly_Average.png')
            
        plt.close()
    else:
        plt.show()
        plt.close()

def center_data_per_commune(df, parametre_info):
    center_records = []
    valid_records = []
    outliers_records = []

    for commune in df['inseecommuneprinc'].unique():
        original_commune_df = df[df['inseecommuneprinc'] == commune]
        entry_count = original_commune_df.shape[0]
        
        if entry_count < 30:
            continue

        # Calculate bounds based on the original data
        temp_mean = original_commune_df['valtraduite'].mean()
        temp_std = original_commune_df['valtraduite'].std()
        lower_bound = temp_mean - 3 * temp_std
        upper_bound = temp_mean + 3 * temp_std
        
        # Separate valid data and outliers
        valid_df = original_commune_df[(original_commune_df['valtraduite'] >= lower_bound) & 
                                        (original_commune_df['valtraduite'] <= upper_bound)]
        outliers_df = original_commune_df[(original_commune_df['valtraduite'] < lower_bound) | 
                                        (original_commune_df['valtraduite'] > upper_bound)]
        
        
        # Compute mean and std on the valid data
        valid_mean = valid_df['valtraduite'].mean()
        valid_std = valid_df['valtraduite'].std()

        '''
        plot_time_series(valid_df, 'valtraduite', outlier_df=outliers_df, mean=valid_mean, std=valid_std, parametre_info=parametre_info, nom_plot='measurement', save=False)
        if entry_count < 30:
            continue
        '''

        # Compute centered reduced data points and add a new column 'centered'
        validcenter_df = valid_df.copy()
        validcenter_df['centered'] = (validcenter_df['valtraduite'] - valid_mean) / valid_std
        center_records.append(validcenter_df)

        valid_records.append(valid_df)
        outliers_records.append(outliers_df)

        

    # Combine all centered data from each commune into one dataframe
    if not center_records:
        return None
    center_df = pd.concat(center_records, ignore_index=True)
    center_df['dateprel'] = pd.to_datetime(center_df['dateprel'])
    center_df.sort_values('dateprel', inplace=True)

    return center_df, valid_records, outliers_records

def analyze_and_plot(df):
    parametre_info_df = df.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()

    for parametre in tqdm(df['cdparametre'].unique(), desc="Processing parameters"):
        '''if parametre in [1303, 1398, 1399, 1340, 1302, 1409, 1301]:
            continue
        '''
        filtered_df = df[df['cdparametre'] == parametre]
        
        parametre_info = parametre_info_df[parametre_info_df['cdparametre'] == parametre].iloc[0]
        result = center_data_per_commune(filtered_df, parametre_info=parametre_info)
        if result is None:
            print(f"No valid data found for {parametre_info['LbCourtParametre']}. Skipping.")
            continue
        center_df, valid_records, outliers_records = result

        valid_mean = center_df['valtraduite'].mean()
        valid_std = center_df['valtraduite'].std()

        # Set the date column as the index to use a time-based rolling window (3 months ≈ 90 days)
        center_df.set_index('dateprel', inplace=True)
        center_df['centered_smoothed'] = center_df['centered'].rolling('60D', center=True).mean()
        center_df['valtraduite_smoothed'] = center_df['valtraduite'].rolling('60D', center=True).mean()
        center_df.reset_index(inplace=True)
        #break
        plot_time_series(center_df, 'valtraduite', mean=valid_mean, std=valid_std, parametre_info=parametre_info, nom_plot='measurement', save=True)
        plot_time_series(center_df, 'valtraduite_smoothed', mean=valid_mean, std=valid_std, parametre_info=parametre_info, nom_plot='smoothed', save=True)
        plot_time_series(center_df, 'centered_smoothed', parametre_info=parametre_info, nom_plot='centered_smoothed', save=True)
        plot_monthly_average(center_df, 'centered_smoothed', parametre_info=parametre_info, save=True)
        
# --- Main Execution ---

if __name__ == '__main__':
    os.makedirs(PLOTS_DIR, exist_ok=True)
    file_path = os.path.join(DATA_DIR, 'merged_data.csv')
    df = load_and_prepare_data(file_path)
    if df.empty:
        print("Filtered DataFrame is empty. Cannot proceed.")
        exit(1)
    #map_all_parametre(df.copy())
    analyze_and_plot(df.copy())
