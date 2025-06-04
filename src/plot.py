import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from parameter_plot import *

def process_file(file):
    df = load_and_prepare_data(file)
    map(df, '17', 'meancommune', save=True)
    map(df, '17', 'stdcommune', save=True)
    
    plot_return = plot_time_series(df, 'deregionalized_valtraduite_smooth', save=True)
    info_n = plot_return if plot_return is not None else (None, None, None)

    slope, mean, std = info_n
    print(f'plot_slope: {slope}, plot_mean: {mean}, plot_std: {std}')
    
    plot_yearly_monthly_average(df, year_col='valtraduite', month_col='centered_reduced_val_smooth', save=True)
    period = calculate_and_plot_fft(df, 'centered_reduced_val_smooth', save=True)
    
    info_d = (None, None, None)
    if period and 350.0 < period < 400.0:
        _, info_d = deperiodize_timeseries(df, save=True)
        return True, info_n, info_d, df['LbCourtParametre'].iloc[0]
    else:
        return False, info_n, info_d, df['LbCourtParametre'].iloc[0]

def create_timeseries(file, file_label_map):
    df = load_and_prepare_data(file)
    df = df.sort_values(by='dateprel').set_index('dateprel')
    return pd.Series(df['centered_reduced_val_smooth'].resample('D').mean().interpolate(method='linear'), name=file_label_map[file])

def calculate_cross_correlation(series_a, series_b):
    result = cross_correlation(series_a, series_b, max_lag=190)
    if result:
        cor, lag, sign = result
        cor, lag, sign = (0 if np.isnan(x) else x for x in (cor, lag, sign))
        return cor, lag, sign
    return 0, 0, 0

def calculate_possible_groups(i, n):
    if n > i or n <= 0:
        return 0
    
    def combinations(n, k):
        if k < 0 or k > n:
            return 0
        if k == 0 or k == n:
            return 1
        if k > n // 2:
            k = n - k
        res = 1
        for j in range(k):
            res = res * (n - j) // (j + 1)
        return res

    return combinations(i, n)

def compute_correlation_matrices(all_files, file_label_map, seasoned=False):
    num_files = len(all_files)
    cor_matrix = np.zeros((num_files, num_files))
    lag_matrix = np.zeros((num_files, num_files))
    sign_matrix = np.zeros((num_files, num_files))
    
    indices = np.random.permutation(num_files)
    mixed_files = [all_files[i] for i in indices]
    
    group_size = 5
    for i in tqdm(range(0, num_files, group_size)):
        dataframes = [create_timeseries(file, file_label_map) for file in mixed_files[i:i + group_size]]
        
        for j, series_a in enumerate(dataframes):
            file_index_a = indices[i + j]
            
            for k in range(j):
                series_b = dataframes[k]
                file_index_b = indices[i + k]
                cor, lag, sign = calculate_cross_correlation(series_a, series_b)
                
                cor_matrix[file_index_a, file_index_b] = cor_matrix[file_index_b, file_index_a] = cor
                lag_matrix[file_index_a, file_index_b] = lag
                lag_matrix[file_index_b, file_index_a] = -lag
                sign_matrix[file_index_a, file_index_b] = sign_matrix[file_index_b, file_index_a] = sign
                
            for m in range(i):
                original_index = indices[m]
                series_b = create_timeseries(all_files[original_index], file_label_map)
                file_index_b = indices[m]
                cor, lag, sign = calculate_cross_correlation(series_a, series_b)
                
                cor_matrix[file_index_a, file_index_b] = cor_matrix[file_index_b, file_index_a] = cor
                lag_matrix[file_index_a, file_index_b] = lag
                lag_matrix[file_index_b, file_index_a] = -lag
                sign_matrix[file_index_a, file_index_b] = sign_matrix[file_index_b, file_index_a] = sign
                
    np.fill_diagonal(cor_matrix, 1.0)
    return cor_matrix, lag_matrix, sign_matrix

def plot_correlation_matrix(cor_matrix, all_files, file_label_map, title):
    plt.figure(figsize=(10, 8))
    plt.imshow(cor_matrix, cmap='RdBu', interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.title(title)
    plt.xticks(range(len(all_files)), [file_label_map[file] for file in all_files], rotation=90)
    plt.yticks(range(len(all_files)), [file_label_map[file] for file in all_files])
    plt.tick_params(axis='both', which='major', labelsize=4)
    plt.tight_layout()
    plt.xlabel('File Index')
    plt.ylabel('File Index')
    plt.savefig(f'plots/{title}.png')

def identify_correlated_groups(cor_matrix, all_files, correlation_threshold=0.7):
    cor_matrix = np.nan_to_num(cor_matrix)
    cor_matrix = np.clip(cor_matrix, -1, 1)
    cor_matrix = (cor_matrix + cor_matrix.T) / 2
    
    clustered_groups = {}
    group_id = 0
    assigned = [False] * len(all_files)
    
    for i in range(len(all_files)):
        if not assigned[i]:
            clustered_groups[group_id] = [i]
            assigned[i] = True
            for j in range(i + 1, len(all_files)):
                if not assigned[j] and cor_matrix[i, j] > correlation_threshold:
                    clustered_groups[group_id].append(j)
                    assigned[j] = True
            group_id += 1
    return clustered_groups

def write_info_to_file(seasoned_files, not_seasoned_files, file_label_map, plot_stats_n, plot_stats_d, all_files):
    with open("plots/info2.txt", "w") as f:
        f.write("Seasoned Files:\n")
        for file in seasoned_files:
            f.write(f"{file}: {file_label_map[file]}\n")
        f.write("\nNot Seasoned Files:\n")
        for file in not_seasoned_files:
            f.write(f"{file}: {file_label_map[file]}\n")
        f.write("\nPlot Statistics:\n")
        for file in all_files:
            f.write(f"{file}:\n")
            f.write(f"  LbCourtParametre: {file_label_map[file]}\n")
            slope_n, mean_n, std_n = plot_stats_n[file]
            slope_d, mean_d, std_d = plot_stats_d[file]
            f.write(f"  Slope, Mean, Std\n")
            f.write(f"  Central Reduced by Communes (n): {int(slope_n)*360}, {mean_n}, {std_n}\n")
            if slope_d is not None:
                f.write(f"  Deseasoned (d):                {int(slope_d)*360}, {mean_d}, {std_d}\n")
            else:
                f.write("  Deseasoned (d):                Not applicable\n")

def main():
    files = load_data_list()
    files = [file.replace(".csv", "_processed.csv") for file in files]
    num_files = len(files)
    
    seasoned_files = []
    not_seasoned_files = []
    file_label_map = {}
    plot_stats_n = {}
    plot_stats_d = {}
    
    for i in tqdm(range(num_files)):
        is_seasoned, plot_stats_n[files[i]], plot_stats_d[files[i]], file_label_map[files[i]] = process_file(files[i])
        if is_seasoned:
            seasoned_files.append(files[i])
        else:
            not_seasoned_files.append(files[i])
    
    if seasoned_files:
        cor_matrix_seasoned, _, _ = compute_correlation_matrices(seasoned_files, file_label_map)
        plot_correlation_matrix(cor_matrix_seasoned, seasoned_files, file_label_map, "Correlation Matrix Seasoned")
    
    if not_seasoned_files:
        cor_matrix_not_seasoned, _, _ = compute_correlation_matrices(not_seasoned_files, file_label_map)
        plot_correlation_matrix(cor_matrix_not_seasoned, not_seasoned_files, file_label_map, "Correlation Matrix Not Seasoned")
    
    all_files = seasoned_files + not_seasoned_files
    write_info_to_file(seasoned_files, not_seasoned_files, file_label_map, plot_stats_n, plot_stats_d, all_files)

if __name__ == "__main__":
    main()