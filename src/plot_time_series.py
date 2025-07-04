import os
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import *

def plot_time_series(df, time_column, value_column, parameter_code, save=False):
    if time_column not in df.columns:
        return

    slope, intercept = linear_regression(df, time_column, value_column)

    df_sorted = df.sort_values(by=time_column).copy()
    df_sorted.set_index(time_column, inplace=True)

    value_mean = df_sorted[value_column].mean()
    value_std = df_sorted[value_column].std()
    
    df_sorted['days_since_start'] = (df_sorted.index - df_sorted.index.min()).days
    df_sorted['regression_line'] = slope * df_sorted['days_since_start'] + intercept

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df_sorted.index, df_sorted[value_column], label=value_column)
    yearly_slope = slope * 365.25
    ax.plot(df_sorted.index, df_sorted['regression_line'], color='purple', linestyle='solid', label=f'Regression Line (Yearly Slope={yearly_slope:.4f})')

    ax.set_xlabel('Date')
    ax.set_ylabel(value_column)
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    ax.axhline(value_mean, color='r', linestyle='-.', label=f'Mean = {value_mean:.2f}')
    ax.axhline(value_mean + value_std, color='g', linestyle='dotted', label=f'Mean + Std = {value_mean + value_std:.2f}')
    ax.axhline(value_mean - value_std, color='g', linestyle='dotted', label=f'Mean - Std = {value_mean - value_std:.2f}')
    ax.legend()
    
    if save:
        plot_dir = os.path.join('plots', f"{df_sorted['LbCourtParametre'].iloc[0]}")
        os.makedirs(plot_dir, exist_ok=True)
        plt.savefig(f'{plot_dir}/centered_reduced.png')
    else:
        plt.show()
    plt.close()
    return (slope, value_mean, value_std)

def main():
    for data_frame in load_folder_data(os.path.join('data', 'clean')):
        parameter_code = data_frame['cdparametre'].iloc[0]
        data_frame['dateprel'] = pd.to_datetime(data_frame['dateprel'], errors='coerce')
        for commune, group_by_commune in get_by_commune(data_frame):
            if len(group_by_commune) < 10:
                continue
            plot_time_series(group_by_commune, 'dateprel', 'valtraduite', parameter_code, save=False)
        break

if __name__ == "__main__":
    main()
