import pandas as pd
import os
import requests
import geopandas as gpd
import time
from tqdm import tqdm
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

GEO_API_URL = 'https://geo.api.gouv.fr/departements/{}/communes?format=geojson&geometry=contour'
MAX_RETRIES = 3
BACKOFF_FACTOR = 2
DEPARTEMENT_CODE = '17'
DATA_DIR = 'data'
PLOTS_DIR = 'plots'

def load_data_list(file_path='data/products/file_row_counts.csv'):
    df = pd.read_csv(file_path)
    df = df[(df['mean'] != 0.0) & (df['zero_percentage'] < 80.0) & (df['rows'] > 10000) & (df['unique_val'] > 10)]
    return df['file_name'].to_list()

def load_and_prepare_data(file_name):
    df = pd.read_csv(file_name)
    df['dateprel'] = pd.to_datetime(df['dateprel'])
    df['inseecommuneprinc'] = df['inseecommuneprinc'].astype(str)
    df['valtraduite'] = pd.to_numeric(df['valtraduite'], errors='coerce').dropna()
    df['cdparametre'] = df['cdparametre'].astype(str)
    df['cdunitereferencesiseeaux'] = df['cdunitereferencesiseeaux'].astype(str)
    df['cddept'] = df['cddept'].astype(str)
    df['LbCourtParametre'] = df['LbCourtParametre'].astype(str)
    return df

def fetch_geodata(departement_code=None):
    if departement_code is None:
        return None
    url = GEO_API_URL.format(departement_code)
    for attempt in range(MAX_RETRIES):
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            gdf = gpd.GeoDataFrame.from_features(data['features'])
            gdf.rename(columns={'code': 'inseecommuneprinc'}, inplace=True)
            gdf['inseecommuneprinc'] = gdf['inseecommuneprinc'].astype(str)
            return gdf
        except requests.exceptions.RequestException as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < MAX_RETRIES - 1:
                time.sleep(BACKOFF_FACTOR ** attempt)
            else:
                print("Max retries reached. Failed to fetch geodata.")
                return None
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return None
    return None

def map(df, departement, col='valtraduite', gdf=None, cdparametre=None, save=False):
    if cdparametre is not None:
        df = df[df['cdparametre'] == cdparametre]
    df = df[df['cddept'] == departement]
    if df.empty:
        print(f"No data available for departement {departement} and parameter {cdparametre}. Exiting.")
        return
    if gdf is None:
        gdf = fetch_geodata(departement_code=departement)
    if gdf is None:
        print("Failed to fetch geodata. Exiting.")
        return
    
    df_grouped = df.groupby('inseecommuneprinc')[col].mean().reset_index()
    df_min = df_grouped[col].min()
    df_max = df_grouped[col].max()
    merged = gdf.merge(df_grouped, on='inseecommuneprinc', how='left')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    merged.plot(
        column=col,
        cmap='viridis',
        linewidth=0.8,
        ax=ax,
        edgecolor='0.8',
        legend=True,
        legend_kwds={'label': f"Value ({df['cdunitereferencesiseeaux'].iloc[0]})", 'orientation': "horizontal"},
        missing_kwds={"color": "lightgrey", "edgecolor": "0.8", "label": "Missing values"},
        vmin=df_min,
        vmax=df_max
    )
    
    ax.set_title(f"{col} of {df['LbCourtParametre'].iloc[0]}", fontsize=15)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    if save:
        path = os.path.join(PLOTS_DIR, f'{df['LbCourtParametre'].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/map_{col}.png')
    else:
        plt.show()
    plt.close()

def plot_yearly_monthly_average(df, year_col='valtraduite', month_col='centered_reduced_val', save=False):
    df['dateprel'] = pd.to_datetime(df['dateprel'])
    df['year'] = df['dateprel'].dt.year
    df['month'] = df['dateprel'].dt.month

    yearly_averages = df.groupby('year')[year_col].mean()
    monthly_averages = df.groupby('month')[month_col].mean()

    total_entries = len(df)
    monthly_entries = df.groupby('month').size()
    yearly_entries = df.groupby('year').size()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 12))

    years_range = range(2019, 2025)
    ax1.plot(yearly_averages.index, yearly_averages.values, marker='o', linestyle='-', color='blue')
    ax1.set_title(f'Yearly Averages of {df["LbCourtParametre"].iloc[0]} \nTotal Entries: {total_entries}')
    ax1.set_ylabel(f'Average Value ({df["cdunitereferencesiseeaux"].iloc[0]})')
    ax1.set_xlabel('Year')
    ax1.grid(axis='y', linestyle='--')
    ax1.set_xticks(years_range)
    ax1.tick_params(axis='x', rotation=0)
    ax1.set_xlim(min(years_range), max(years_range))
    for year, count in yearly_entries.items():
        ax1.annotate(f'Entries: {count}', xy=(year, yearly_averages[year]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    colors = ['green' if x >= 0 else 'red' for x in monthly_averages]
    ax2.bar(monthly_averages.index, monthly_averages.values, color=colors)
    ax2.set_title(f'Monthly Averages of {df["LbCourtParametre"].iloc[0]}\nTotal Entries: {total_entries}')
    ax2.set_ylabel('Average Centered Reduced Value (No Unit)')
    ax2.set_xlabel('Month')
    ax2.set_xticks(monthly_averages.index)
    ax2.tick_params(axis='x', rotation=0)
    ax2.grid(axis='y', linestyle='--')
    for month, count in monthly_entries.items():
        ax2.annotate(f'Entries: {count}', xy=(month, monthly_averages[month]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()

    if save:
        path = os.path.join(PLOTS_DIR, f'{df["LbCourtParametre"].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/Yearly_Monthly_Averages.png')
    else:
        plt.show()
    plt.close()

def linear_regression(df, time_col, value_col):
    time = (df[time_col] - df[time_col].min()).dt.days.values.reshape(-1, 1)
    values = df[value_col].values

    model = LinearRegression()
    model.fit(time, values)

    slope = model.coef_[0]
    intercept = model.intercept_

    return slope, intercept

def plot_time_series(df, col='valtraduite', save=False):
    mean = df[col].mean()
    std = df[col].std()
    df = df.sort_values(by='dateprel')
    
    slope, intercept = linear_regression(df, 'dateprel', col)
    
    df['time_num'] = (df['dateprel'] - df['dateprel'].min()).dt.days
    df['regression_line'] = slope * df['time_num'] + intercept

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['dateprel'], df[col], label=col)
    yearly_slope = slope * 365.25
    ax.plot(df['dateprel'], df['regression_line'], color='purple', linestyle='solid', label=f'Regression Line (Yearly Slope={yearly_slope:.4f})')

    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    ax.axhline(mean, color='r', linestyle='-.', label=f'Mean = {mean:.2f}')
    ax.axhline(mean + std, color='g', linestyle='dotted', label=f'Mean + Std = {mean + std:.2f}')
    ax.axhline(mean - std, color='g', linestyle='dotted', label=f'Mean - Std = {mean - std:.2f}')
    ax.legend()
    
    if save:
        path = os.path.join(PLOTS_DIR, f'{df['LbCourtParametre'].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/centered_reduced.png')
    else:
        plt.show()
    plt.close()
    return (slope, mean, std)

def smooth(df, window_size=60, col='centered_reduced_val'):
    if 'dateprel' not in df.columns:
        print("Error: 'dateprel' column is missing in the DataFrame.")
        return df
    df = df.sort_values(by='dateprel').set_index('dateprel')
    df[f'{col}_smooth'] = df[col].rolling(f'{window_size}D', center=True, min_periods=1).mean()
    df = df.reset_index()
    return df

def remove_outliers(df, col='valtraduite'):
    mean = df[col].mean()
    std = df[col].std()
    lower_bound = mean - 2 * std
    upper_bound = mean + 2 * std
    return df[(df[col] >= lower_bound) & (df[col] <= upper_bound)]

def center_reduce(df):
    df['centered_reduced_val'] = np.nan

    def process_commune(valtraduite):
        if len(valtraduite) < 50:
            return None
        
        mean = valtraduite.mean()
        std = valtraduite.std()
        
        mask = (valtraduite >= mean - 2 * std) & (valtraduite <= mean + 2 * std)
        filtered_valtraduite = valtraduite[mask]
        
        if filtered_valtraduite.empty:
            return None
        
        centered_reduced_val = (filtered_valtraduite - mean) / std
        return centered_reduced_val.index, centered_reduced_val.values

    grouped = df.groupby('inseecommuneprinc')['valtraduite']

    for commune, valtraduite in grouped:
        result = process_commune(valtraduite)
        if result is not None:
            indices, values = result
            df.loc[indices, 'centered_reduced_val'] = values
    
    df = df.dropna(subset=['centered_reduced_val'])

    return df

def calculate_and_plot_fft(pdf, column='deregionalized_valtraduite_smooth', save=False):
    if column not in pdf.columns:
        return None

    pdf = pdf.sort_values(by='dateprel').set_index('dateprel')
    daily_avg = pdf[column].resample('D').mean().interpolate(method='linear')
    if daily_avg.isnull().all():
        return None

    daily_avg_no_nan = daily_avg.dropna()
    fft = np.fft.fft(daily_avg_no_nan.values)
    frequencies = np.fft.fftfreq(len(daily_avg_no_nan))
    abs_fft = np.abs(fft)
    max_index = min(len(abs_fft) // 2, int(len(daily_avg_no_nan) / 100))
    dominant_frequency_index = np.argmax(abs_fft[1:max_index]) + 1
    dominant_frequency = frequencies[dominant_frequency_index]
    period = 1 / abs(dominant_frequency) if dominant_frequency != 0 else np.nan
    magnitude_threshold = 0.8
    peaks_above_threshold = np.where(abs_fft[1:max_index] > (abs_fft[dominant_frequency_index] * magnitude_threshold))[0] + 1
    if len(peaks_above_threshold) > 1:
        period = None

    positive_frequencies = frequencies[1:len(abs_fft)//2]
    periods = 1 / np.abs(positive_frequencies)
    magnitude_threshold = np.mean(abs_fft[1:len(abs_fft)//2]) * 2
    
    plt.figure(figsize=(12, 6))
    if period is None or np.isnan(period) or period > 365 * 2 or abs_fft[dominant_frequency_index] < magnitude_threshold:
        plt.plot(periods, abs_fft[1:len(abs_fft)//2], label='No significant frequency found')
    else:
        plt.plot(periods, abs_fft[1:len(abs_fft)//2], label=f'Dominant Period: {period:.2f} days')
    
    plt.xlabel('Period (days)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Spectrum of {pdf["LbCourtParametre"].iloc[0]}')
    plt.grid(True, which='major', axis='x', linewidth=0.7, color='gray')
    plt.xticks(np.arange(0, 365 * 2, 365/4), [f'{x:.0f}' for x in np.arange(0, 365 * 2, 365/4)])
    plt.xlim(0, 365 * 2)
    plt.legend()
    plt.tight_layout()
    
    if save:
        path = os.path.join(PLOTS_DIR, f'{pdf["LbCourtParametre"].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/fft_spectrum.png')
    else:
        plt.show()
    plt.close()

    return period

def deperiodize_timeseries(pdf, save=False):
    column = 'deregionalized_valtraduite_smooth'
    if column not in pdf.columns:
        return None, (None, None, None)

    pdf = pdf.sort_values(by='dateprel').set_index('dateprel')
    daily_avg = pdf[column].resample('D').mean().interpolate(method='linear')
    if daily_avg.isnull().all():
        return None, (None, None, None)

    daily_avg_no_nan = daily_avg.dropna()
    fft = np.fft.fft(daily_avg_no_nan.values)
    abs_fft = np.abs(fft)
    dominant_frequency_index = np.argmax(abs_fft[1:len(abs_fft)//2]) + 1
    fft_filtered = fft.copy()
    num_harmonics = 20

    for i in range(1, num_harmonics + 1):
        harmonic_index = dominant_frequency_index * i
        if harmonic_index < len(fft_filtered) // 2:
            fft_filtered[harmonic_index] = (fft_filtered[harmonic_index - 1] + fft_filtered[harmonic_index + 1]) / 2
            fft_filtered[-harmonic_index] = (fft_filtered[-harmonic_index - 1] + fft_filtered[-harmonic_index + 1]) / 2

    ifft = np.fft.ifft(fft_filtered)

    slope, intercept = linear_regression(pd.DataFrame({'dateprel': daily_avg_no_nan.index, 'valtraduite': ifft.real}).set_index('dateprel').reset_index(), 'dateprel', 'valtraduite')
    mean = ifft.real.mean()
    std = ifft.real.std()
    
    time_num = (daily_avg_no_nan.index - daily_avg_no_nan.index.min()).days
    regression_line = slope * time_num + intercept

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(daily_avg_no_nan.index, ifft.real, label='Deperiodized Time Series')
    yearly_slope = slope * 365.25
    ax.plot(daily_avg_no_nan.index, regression_line, color='purple', linestyle='solid', label=f'Regression Line (Yearly Slope={yearly_slope:.4f})')

    ax.set_xlabel('Date')
    ax.set_ylabel(f'Value ({pdf["cdunitereferencesiseeaux"].iloc[0]})')
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    ax.axhline(mean, color='r', linestyle='-.', label=f'Mean = {mean:.2f}')
    ax.axhline(mean + std, color='g', linestyle='dotted', label=f'Mean + Std = {mean + std:.2f}')
    ax.axhline(mean - std, color='g', linestyle='dotted', label=f'Mean - Std = {mean - std:.2f}')
    ax.legend()
    ax.set_title(f'Deseasoned Time Series of {pdf["LbCourtParametre"].iloc[0]}')
    
    if save:
        path = os.path.join(PLOTS_DIR, f'{pdf["LbCourtParametre"].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/deseasoned_time_series.png')
    else:
        plt.show()
    plt.close()

    return pd.Series(ifft.real, index=daily_avg_no_nan.index), (slope, mean, std)

def cross_correlation(series_a: pd.Series,
                      series_b: pd.Series,
                      max_lag: int = 30) -> pd.Series:
    """
    Calcule la corrélation croisée entre deux séries temporelles
    pour des lags de -max_lag à +max_lag.
    Retourne une Series dont l’index sont les lags et les valeurs
    les coefficients de corrélation.
    """
    # on aligne les deux séries sur l’intersection des dates
    df = pd.concat([series_a, series_b], axis=1).dropna()
    if df.empty:
        return np.nan, np.nan, np.nan
    a, b = df.iloc[:,0], df.iloc[:,1]
    lags = range(-max_lag, max_lag+1)
    corrs = []
    for lag in lags:
        if lag < 0:
            corr = a.shift(-lag).corr(b)
        else:
            corr = a.corr(b.shift(lag))
        corrs.append(corr)
    
    cross_corr = pd.Series(corrs, index=lags)
    abs_cross_corr = abs(cross_corr).fillna(0)
    peak_value = abs_cross_corr.max()
    peak_lag = abs_cross_corr.idxmax()
    peak_sign = cross_corr.fillna(0)[peak_lag] >= 0 if not np.isnan(peak_lag) else np.nan
    return peak_value, peak_lag, peak_sign