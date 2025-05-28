import pandas as pd
from parameter_plot import *
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import AgglomerativeClustering
import numpy as np


def main():
    files = load_data_list()
    files = [file.replace(".csv", "_processed.csv") for file in files]
    
    num_files = len(files)-45
    cor_matrix = np.zeros((num_files, num_files))
    lag_matrix = np.zeros((num_files, num_files))
    sign_matrix = np.zeros((num_files, num_files))
    seasoned_values = []

    for i in tqdm(range(num_files - 1, -1, -1)):
        df_a = load_and_prepare_data(files[i])
        map(df_a, '17', 'meancommune', save=True)
        map(df_a, '17', 'stdcommune', save=True)
        plot_time_series(df_a, 'deregionalized_valtraduite_smooth', save=True)
        plot_yearly_monthly_average(df_a, year_col='valtraduite', month_col='centered_reduced_val_smooth', save=True)
        period = calculate_and_plot_fft(df_a, 'centered_reduced_val_smooth', save=True)
        if period is not None:
            print(period)
            print(350.0 < period and period < 400.0)
            if 350.0 < period and period < 400.0:
                dep_a = deperiodize_timeseries(df_a, save=True)
                seasoned_values.append(df_a['LbCourtParametre'].iloc[0])
        

        df_a = df_a.sort_values(by='dateprel').set_index('dateprel')
        avg_a = df_a['centered_reduced_val_smooth'].resample('D').mean().interpolate(method='linear')
        series_a = pd.Series(avg_a, name='series_a')

        for j in tqdm(range(i), leave=False):
            df_b = load_and_prepare_data(files[j])
            df_b = df_b.sort_values(by='dateprel').set_index('dateprel')
            avg_b = df_b['centered_reduced_val_smooth'].resample('D').mean().interpolate(method='linear')    
            series_b = pd.Series(avg_b, name='series_b')
            
            result = cross_correlation(series_a, series_b, max_lag=190)
            if result is not None:
                cor, lag, sign = result
            else:
                cor, lag, sign = 0, 0, 0
            if np.isnan(cor):
                cor = 0
            if np.isnan(lag):
                lag = 0
            if np.isnan(sign):
                sign = 0
            cor_matrix[i, j] = cor
            cor_matrix[j, i] = cor
            lag_matrix[i, j] = lag
            lag_matrix[j, i] = -lag
    sign_matrix[i, j] = sign
    sign_matrix[j, i] = sign
    cor_matrix[i==j] = 1.0
    plt.figure(figsize=(10, 8))
    plt.imshow(cor_matrix, cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title('Correlation Matrix')
    plt.xticks(range(num_files), [file.split('_')[0] for file in files[:num_files]], rotation=90)
    plt.yticks(range(num_files), [file.split('_')[0] for file in files[:num_files]])
    plt.tick_params(axis='both', which='major', labelsize=4)
    plt.tight_layout()
    plt.xlabel('File Index')
    plt.ylabel('File Index')
    plt.savefig('plots/correlation_matrix.png')
    # Ensure the correlation matrix is symmetric and has no NaN values
    cor_matrix = np.nan_to_num(cor_matrix)
    cor_matrix = np.clip(cor_matrix, -1, 1)  # Ensure values are within [-1, 1]
    cor_matrix = (cor_matrix + cor_matrix.T) / 2  # Ensure symmetry

    # Define a correlation threshold
    correlation_threshold = 0.7  # Adjust as needed

    # Identify groups of highly correlated files
    clustered_groups = {}
    group_id = 0
    assigned = [False] * num_files

    for i in range(num_files):
        if not assigned[i]:
            clustered_groups[group_id] = [i]
            assigned[i] = True
            for j in range(i + 1, num_files):
                if not assigned[j] and cor_matrix[i, j] > correlation_threshold:
                    clustered_groups[group_id].append(j)
                    assigned[j] = True
            group_id += 1

    print("Clustered Groups:", clustered_groups)
        

if __name__ == "__main__":
    main()


