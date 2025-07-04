import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from Utilities import *
from tqdm import tqdm

PLOTS_DIR = 'plots'

def plot_yearly_monthly_average(df, year_col='valtraduite', month_col='centered_reduced_val', save=False):
    if 'dateprel' not in df.columns:
        return None
    df['dateprel'] = pd.to_datetime(df['dateprel'])
    df['year'] = df['dateprel'].dt.year
    df['month'] = df['dateprel'].dt.month

    parameter_code = df['cdparametre'].iloc[0]
    yearly_means = df.groupby('year')[year_col].mean()
    monthly_means = df.groupby('month')[month_col].mean()
    total_data_points = len(df)
    monthly_counts = df.groupby('month').size()
    yearly_counts = df.groupby('year').size()

    fig, (ax_year, ax_month) = plt.subplots(2, 1, figsize=(12, 12))
    years_range = range(2016, 2025)
    ax_year.plot(yearly_means.index, yearly_means.values, marker='o', linestyle='-', color='blue')
    ax_year.set_title(f'Yearly Averages of {get_cdparam_name(parameter_code)} \nTotal Entries: {total_data_points}')
    ax_year.set_ylabel('Average Value')
    ax_year.set_xlabel('Year')
    ax_year.grid(axis='y', linestyle='--')
    ax_year.set_xticks(years_range)
    ax_year.set_xlim(min(years_range), max(years_range))
    for year, count in yearly_counts.items():
        ax_year.annotate(f'Entries: {count}', xy=(year, yearly_means[year]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    bar_colors = ['green' if x >= 0 else 'red' for x in monthly_means]
    ax_month.bar(monthly_means.index, monthly_means.values, color=bar_colors)
    ax_month.set_title(f'Monthly Averages of {get_cdparam_name(parameter_code)}\nTotal Entries: {total_data_points}')
    ax_month.set_ylabel('Average Centered Reduced Value')
    ax_month.set_xlabel('Month')
    ax_month.set_xticks(monthly_means.index)
    ax_month.grid(axis='y', linestyle='--')
    for month, count in monthly_counts.items():
        ax_month.annotate(f'Entries: {count}', xy=(month, monthly_means[month]), xytext=(0, 5), textcoords='offset points', ha='center', fontsize=8)

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(os.getcwd(), 'plots', 'yearmonth', f'{parameter_code}_yearmonth.png'))
    else:
        plt.show()
    plt.close()

    if not monthly_means.empty:
        return monthly_means.idxmax()
    return None

def linear_regression(df, time_col, value_col):
    df = df.copy()
    df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
    df = df.dropna(subset=[time_col, value_col])
    days_since_start = (df[time_col] - df[time_col].min()).dt.days.values.reshape(-1, 1)
    values = df[value_col].values
    model = LinearRegression()
    model.fit(days_since_start, values)
    return model.coef_[0], model.intercept_

def plot_time_series(df, col='valtraduite', save=False):
    df = df.copy()
    if 'dateprel' not in df.columns:
        return None
    df['dateprel'] = pd.to_datetime(df['dateprel'], errors='coerce')
    df = df.dropna(subset=['dateprel', col])
    df = df[df['dateprel'] >= pd.Timestamp('2017-01-01')]
    if df.empty:
        return None
    value_mean = df[col].mean()
    value_std = df[col].std()
    df = df.sort_values(by='dateprel')
    parameter_code = df['cdparametre'].iloc[0]
    slope, intercept = linear_regression(df, 'dateprel', col)
    df['days_since_start'] = (df['dateprel'] - df['dateprel'].min()).dt.days
    df['regression_line'] = slope * df['days_since_start'] + intercept

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(df['dateprel'], df[col], label=col)
    yearly_slope = slope * 365.25
    ax.plot(df['dateprel'], df['regression_line'], color='purple', linestyle='solid', label=f'Regression Line (Yearly Slope={yearly_slope:.4f})')
    ax.set_title(f'Time Series of {get_cdparam_name(parameter_code)}\nSlope: {slope:.4f} per day, Mean: {value_mean:.2f}, Std: {value_std:.2f}')
    ax.set_xlabel('Date')
    ax.set_ylabel(col)
    ax.grid()
    ax.xaxis.set_major_locator(plt.matplotlib.dates.YearLocator())
    ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%Y'))
    plt.tight_layout()
    ax.axhline(value_mean, color='r', linestyle='-.', label=f'Mean = {value_mean:.2f}')
    ax.axhline(value_mean + value_std, color='g', linestyle='dotted', label=f'Mean + Std = {value_mean + value_std:.2f}')
    ax.axhline(value_mean - value_std, color='g', linestyle='dotted', label=f'Mean - Std = {value_mean - value_std:.2f}')
    ax.legend()
    if save:
        plt.savefig(os.path.join(os.getcwd(), 'plots', 'time', 'raw', f'{parameter_code}_time_series.png'))
    else:
        plt.show()
    plt.close()
    return (slope, value_mean, value_std)
def main():
    months_with_highest_value = {}
    slopes = {}

    for df in tqdm(load_folder_data(os.path.join(os.getcwd(), 'data', 'upgraded')), desc='Processing Data', total=139):
        plot_time_series(df, col='valtraduite_S', save=True)
        result = plot_time_series(df, col='valtraduite_S', save=True)
        if result is not None:
            slope, mean, std = result
            slopes[df['cdparametre'].iloc[0]] = slope
        else:
            if 'cdparametre' in df.columns:
                slopes[df['cdparametre'].iloc[0]] = None
            continue
        month_with_highest_value = plot_yearly_monthly_average(df, year_col='valtraduite', month_col='valtraduite_D', save=True)
        if 'cdparametre' in df.columns and month_with_highest_value is not None:
            cdparam = df['cdparametre'].iloc[0]
            months_with_highest_value[cdparam] = month_with_highest_value

    with open(os.path.join(os.getcwd(), 'months_with_highest_value.txt'), 'w') as f:
        for cdparam, month in months_with_highest_value.items():
            f.write(f"{cdparam}: {month}\n")
    with open(os.path.join(os.getcwd(), 'slopes.txt'), 'w') as f:
        for cdparam, slope in slopes.items():
            if slope is not None:
                f.write(f"{cdparam}: {slope}\n")
            else:
                f.write(f"{cdparam}: None\n")

if __name__ == "__main__":
    main()
