import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from Utilities import *
from tqdm import tqdm
from scipy.signal import find_peaks

DEPARTEMENT_CODE = '17'
DATA_DIR = 'data'
PLOTS_DIR = 'plots'

def calculate_and_plot_fft(dataframe, column='deregionalized_valtraduite_smooth', save=False):
    if column not in dataframe.columns:
        return None, None

    dataframe = dataframe.sort_values(by='dateprel').set_index('dateprel')
    daily_mean_series = dataframe[column].resample('D').mean().interpolate(method='linear')
    if daily_mean_series.isnull().all():
        return None, None
    parameter_code = int(dataframe['cdparametre'].iloc[0])
    daily_mean_no_nan = daily_mean_series.dropna()

    fft_values = np.fft.fft(daily_mean_no_nan.values)
    fft_frequencies = np.fft.fftfreq(len(daily_mean_no_nan), d=1)

    fft_magnitude = np.abs(fft_values)
    half_magnitude = fft_magnitude[1:len(fft_magnitude)//2]
    half_frequencies = fft_frequencies[1:len(fft_magnitude)//2]

    peak_indices, peak_properties = find_peaks(half_magnitude, height=np.mean(half_magnitude) * 2)

    if len(peak_indices) > 0:
        dominant_peak_index = peak_indices[np.argmax(peak_properties['peak_heights'])]
        dominant_frequency = half_frequencies[dominant_peak_index]
        dominant_period = 1 / abs(dominant_frequency) if dominant_frequency != 0 else np.nan
    else:
        dominant_period = None

    if len(peak_indices) > 0:
        dominant_peak_index = peak_indices[np.argmax(peak_properties['peak_heights'])]
        dominant_frequency_index = dominant_peak_index + 1
        dominant_frequency = fft_frequencies[dominant_frequency_index]
        dominant_period = 1 / abs(dominant_frequency) if dominant_frequency != 0 else np.nan
    else:
        dominant_period = None

    magnitude_threshold = np.mean(fft_magnitude[1:len(fft_magnitude)//2]) * 2
    max_index = len(fft_magnitude)//2
    peaks_above_threshold = np.where(fft_magnitude[1:max_index] > (fft_magnitude[dominant_frequency_index] * magnitude_threshold))[0] + 1
    if len(peaks_above_threshold) > 1:
        dominant_period = None

    positive_frequencies = fft_frequencies[1:len(fft_magnitude)//2]
    periods = 1 / np.abs(positive_frequencies)
    peak_periods = periods[peak_indices] if len(peak_indices) > 0 else np.array([])

    plt.figure(figsize=(12, 6))
    if dominant_period is None or np.isnan(dominant_period) or dominant_period > 365 * 2 or fft_magnitude[dominant_frequency_index] < magnitude_threshold:
        plt.plot(periods, fft_magnitude[1:len(fft_magnitude)//2], label='No significant frequency found')
    else:
        plt.plot(periods, fft_magnitude[1:len(fft_magnitude)//2], label=f'Dominant Period: {dominant_period:.2f} days')

    plt.xlabel('Period (days)')
    plt.ylabel('Magnitude')
    plt.title(f'FFT Spectrum of {get_cdparam_name(parameter_code)}, Centered reduced by commune')
    plt.grid(True, which='major', axis='x', linewidth=0.7, color='gray')
    plt.xticks(np.arange(0, 365 * 2, 365/4), [f'{x:.0f}' for x in np.arange(0, 365 * 2, 365/4)])
    plt.xlim(0, 365 * 2)
    plt.legend()
    plt.tight_layout()

    if save:
        plt.savefig(os.path.join(PLOTS_DIR, 'period', f'{parameter_code}.png'))
    else:
        plt.show()
    plt.close()

    return dominant_period, peak_periods.tolist()

def main():
    period_peaks_by_param = {}

    for dataframe in tqdm(load_folder_data(os.path.join(os.getcwd(), 'data', 'upgraded')), desc="Processing data", total=139):
        parameter_code = int(dataframe['cdparametre'].iloc[0])
        if 'dateprel' in dataframe.columns:
            dataframe['dateprel'] = pd.to_datetime(dataframe['dateprel'], errors='coerce')
            _, peak_periods = calculate_and_plot_fft(dataframe, column='valtraduite_DS', save=True)
            period_peaks_by_param[parameter_code] = peak_periods

    with open('all_peaks.txt', 'w') as file:
        for parameter_code, peaks in period_peaks_by_param.items():
            file.write(f"{parameter_code}: {peaks}\n")

if __name__ == "__main__":
    main()
