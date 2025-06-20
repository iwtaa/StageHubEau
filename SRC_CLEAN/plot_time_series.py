import os
import pandas as pd
import matplotlib.pyplot as plt
from Utilities import *

def plot_time_series(df, timecol, valuecol, cdparam, save=False):
    if timecol not in df.columns:
        print(f"Error: The column '{timecol}' does not exist in the DataFrame.")
        return

    slope, intercept = linear_regression(df, timecol, valuecol)

    df.sort_values(by=timecol, inplace=True)
    df.set_index(timecol, inplace=True)

    mean = df[valuecol].mean()
    std = df[valuecol].std()
    
    df['time_num'] = (df.index - df.index.min()).days
    df['regression_line'] = slope * df['time_num'] + intercept

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df.index, df[valuecol], label=valuecol)
    yearly_slope = slope * 365.25
    ax.plot(df.index, df['regression_line'], color='purple', linestyle='solid', label=f'Regression Line (Yearly Slope={yearly_slope:.4f})')

    ax.set_xlabel('Date')
    ax.set_ylabel(valuecol)
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    ax.axhline(mean, color='r', linestyle='-.', label=f'Mean = {mean:.2f}')
    ax.axhline(mean + std, color='g', linestyle='dotted', label=f'Mean + Std = {mean + std:.2f}')
    ax.axhline(mean - std, color='g', linestyle='dotted', label=f'Mean - Std = {mean - std:.2f}')
    ax.legend()
    
    if save:
        path = os.path.join('plots', f'{df['LbCourtParametre'].iloc[0]}')
        os.makedirs(path, exist_ok=True)
        plt.savefig(f'{path}/centered_reduced.png')
    else:
        plt.show()
    plt.close()
    return (slope, mean, std)

def main():
    for df in load_folder_data(os.path.join('data', 'clean')):
        cdparam = df['cdparametre'].iloc[0]
        df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
        for commune, df_group in get_by_commune(df):
            if len(df_group) < 10:
                #print(f"Skipping {commune} due to insufficient data points.")
                continue
            print(df_group.head())
            print(len(df_group))
            plot_time_series(df_group, 'dateprel', 'valtraduite', cdparam, save=False)
            
        break

    return

if __name__ == "__main__":
    main()