# AUTOCORRELATION

from processing_functions import *
from parameter_plot import *

import pandas as pd
import itertools
from tqdm import tqdm
import numpy as np

df = load_and_prepare_data('data/merged_data.csv', debug=False)

parametre_info_df = df.groupby('cdparametre').agg({'cdunitereferencesiseeaux': 'first', 'LbCourtParametre': 'first'}).reset_index()
parametre_center_df = {}
#df = df[df['cdparametre'].isin([1963, 6355])]
for parametre in tqdm(df['cdparametre'].unique(), desc="Processing parameters"):
    df_filtered = df[df['cdparametre'] == parametre]
    parametre_info = parametre_info_df[parametre_info_df['cdparametre'] == parametre].iloc[0]
    
    temp, _, _ = center_data_per_commune(df_filtered, parametre_info=parametre_info)
    if temp is not None:
        parametre_center_df[parametre] = temp
    

import matplotlib.pyplot as plt
'''
for parametre1, parametre2 in itertools.combinations(parametre_center_df.keys(), 2):
    print(parametre1, parametre2)
    df1 = parametre_center_df[parametre1][['dateprel', 'centered']].set_index('dateprel')
    df2 = parametre_center_df[parametre2][['dateprel', 'centered']].set_index('dateprel')
    
    # Ensure the index is unique by grouping and taking the mean
    df1 = df1.groupby('dateprel').mean()
    df2 = df2.groupby('dateprel').mean()
    
    # Use pd.concat for efficient merging and alignment
    df_merged = pd.concat([df1, df2], axis=1, keys=['param1', 'param2']).dropna()
    
    # Convert to numpy arrays for faster calculations
    centered_1 = df_merged['param1', 'centered'].to_numpy()
    centered_2 = df_merged['param2', 'centered'].to_numpy()
    
    # Calculate cross-correlation using numpy's correlate function
    cross_correlation = np.correlate(centered_1, centered_2, mode='full')
    
    # Determine the lag axis
    lags = np.arange(-len(centered_1) + 1, len(centered_1))

    # Normalize the cross-correlation
    cross_correlation = cross_correlation / np.sqrt(np.sum(centered_1**2) * np.sum(centered_2**2))
    
    plt.plot(lags, cross_correlation)
    plt.xlabel('Lag')
    plt.ylabel('Cross-correlation')
    plt.title(f'Cross-correlation between {parametre1} and {parametre2}')
    plt.show()
'''


def center_data_per_commune(df):
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
        return None, None, None
    center_df = pd.concat(center_records, ignore_index=True)
    center_df['dateprel'] = pd.to_datetime(center_df['dateprel'])
    center_df.sort_values('dateprel', inplace=True)
    return center_df, valid_records, outliers_records

for parametre in parametre_center_df.keys():
    print(parametre)
    df1 = parametre_center_df[parametre][['dateprel', 'centered']].set_index('dateprel')
    
    # Ensure the index is unique by grouping and taking the mean
    df1 = df1.groupby('dateprel').mean()
    
    # Convert to numpy arrays for faster calculations
    centered_1 = df1['centered'].to_numpy()
    
    # Calculate cross-correlation using numpy's correlate function
    autocorrelation = np.correlate(centered_1, centered_1, mode='full')
    
    # Determine the lag axis
    lags = np.arange(-len(centered_1) + 1, len(centered_1))

    # Normalize the cross-correlation
    autocorrelation = autocorrelation / np.sum(centered_1**2)
    
    plt.plot(lags, autocorrelation)
    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    plt.title(f'Autocorrelation for {parametre}')
    plt.show()



















#PLOT ANALYZE 2

import pandas as pd
import os
import requests
import geopandas as gpd
import time
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

def map(df, departement, cdparametre=None, debug=False):
    if cdparametre is not None:
        df = df[df['cdparametre'] == cdparametre]
    df = df[df['cddept'] == departement]
    if len(df) == 0:
        print(f"No data available for departement {departement} and parameter {cdparametre}. Exiting.")
        return
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

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)
    print("Loading and preparing data...")
    file_names = load_data_list()
    for file_name in file_names:
        print(file_name)
        df = load_and_prepare_data(file_name, debug=True)
        map(df, DEPARTEMENT_CODE, debug=True)
        

if __name__ == "__main__":
    main()