import pandas as pd
from Utilities import load_folder_data, get_by_commune, get_cdparam_name
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

communes_csv_path = os.path.join(os.getcwd(), 'data', 'communes-france-2024.csv')
communes_df = pd.read_csv(communes_csv_path)
insee_to_population = dict(zip(communes_df['code_insee'], communes_df['population']))

par_sandre_path = os.path.join(os.getcwd(), 'data', 'PAR_SANDRE_short.txt')
par_sandre_df = pd.read_csv(par_sandre_path, sep='\t')

clean_data_folder = os.path.join(os.getcwd(), 'data', 'clean')
plots_output_folder = os.path.join(os.getcwd(), 'plots', 'pop')

for data_frame in tqdm(load_folder_data(clean_data_folder), desc="Processing files", total=139):
    parameter_code = data_frame['cdparametre'].iloc[0]
    population_values = []
    translated_values = []
    for insee_code, commune_group in get_by_commune(data_frame):
        if insee_code not in insee_to_population:
            continue
        population = insee_to_population[insee_code]
        values = commune_group['valtraduite'].values
        population_values.extend([population] * len(values))
        translated_values.extend(values)
    plt.scatter(population_values, translated_values, s=0.5, alpha=0.2, label='Data')
    plt.title(f'Value per Population of {get_cdparam_name(parameter_code)}')
    plt.xlabel('Population')
    plt.ylabel('Mean Value')
    plt.xscale('log')
    plt.legend(loc='upper right', markerscale=10, fontsize='small')
    if translated_values:
        lower_quantile = pd.Series(translated_values).quantile(0.01)
        upper_quantile = pd.Series(translated_values).quantile(0.99)
        if lower_quantile == upper_quantile:
            lower_quantile = min(translated_values)
            upper_quantile = max(translated_values)
        plt.ylim(lower_quantile, upper_quantile)
    if len(population_values) > 1:
        log_population = np.log10(np.array(population_values)).reshape(-1, 1)
        translated_array = np.array(translated_values)
        histogram_counts, bin_edges = np.histogram(log_population, bins=20)
        bin_indices = np.digitize(log_population.flatten(), bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, len(histogram_counts) - 1)
        histogram_counts = np.where(histogram_counts == 0, 1, histogram_counts)
        weights = 1.0 / histogram_counts[bin_indices]
        regression_model = LinearRegression()
        regression_model.fit(log_population, translated_array, sample_weight=weights)
        fit_population = np.linspace(min(population_values), max(population_values), 100)
        fit_values = regression_model.predict(np.log10(fit_population).reshape(-1, 1))
        plt.plot(fit_population, fit_values, color='red', linewidth=2, label='Weighted Linear Regression')
        plt.legend()
    plot_filename = os.path.join(plots_output_folder, f'{parameter_code}_population.png')
    plt.savefig(plot_filename, bbox_inches='tight')
    plt.close()
