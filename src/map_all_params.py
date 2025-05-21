import pandas as pd
import json
from parameter_plot_charente_maritime import *
'''    
file_path = 'DIS_RESULT_17_2024.txt'
df = pd.read_csv(file_path, sep=',', encoding='latin-1')
df['cdparametre'] = df['cdparametre'].astype(str).str[:-2]

with open('parameters.json', 'r') as f:
    params2 = {
        p['CdParametre']
        for p in json.load(f)['REFERENTIELS']['Referentiel']['Parametre']
        if p.get('TypeParametre') == '2'
    }

cdparametre_values = df['cdparametre']
filtered_counts = cdparametre_values[cdparametre_values.isin(params2)].value_counts()
keys_array = filtered_counts[filtered_counts >= 300].index.tolist()

df = df[df['cdparametre'].isin(keys_array)]
for code in df['cdparametre'].unique():
'''

df_plv = pd.read_csv('DIS_PLV_2024.txt', sep=',', encoding='latin-1')
df_result = pd.read_csv('DIS_RESULT_2024.txt', sep=',', encoding='latin-1')

df_plv_17 = df_plv[df_plv['cddept'] == 17]
df_result_17 = df_result[df_result['cddept'] == 17]

df_plv_17 = df_plv_17[['referenceprel', 'inseecommuneprinc']]
df_result_17 = df_result_17[['referenceprel', 'cdparametre', 'valtraduite', 'cdunitereferencesiseeaux']]

df_merged = pd.merge(df_result_17, df_plv_17, on='referenceprel', how='inner')
df_merged['cdparametre'] = df_merged['cdparametre'].astype(str).str[:-2]
print(df_merged.head())

cdparametre_counts = df_merged['cdparametre'].value_counts()
valid_cdparametre = cdparametre_counts[cdparametre_counts >= 250].index
df_merged = df_merged[df_merged['cdparametre'].isin(valid_cdparametre)]
cdparametre_counts = df_merged['cdparametre'].value_counts()
print("Occurrences of each cdparametre value:")
print(cdparametre_counts)

df_merged.dropna(inplace=True)
null_valtraduite_counts = df_merged['valtraduite'].isnull().sum()
print("\nNumber of null values in valtraduite:", null_valtraduite_counts)

with open('parameters.json', 'r') as f:
    params = json.load(f)['REFERENTIELS']['Referentiel']['Parametre']
    

departement_code = '17'
date_max_prelevement = '2025-01-01 00:00:00'
date_min_prelevement = '2024-01-01 00:00:00'

gdf = fetch_geodata(departement_code)

from APIcalls import getNomParametre
from tqdm import tqdm
min_by_parametre = df_merged.groupby('cdparametre')['valtraduite'].min()
max_by_parametre = df_merged.groupby('cdparametre')['valtraduite'].max()
min_by_parametre = min_by_parametre.to_dict()
max_by_parametre = max_by_parametre.to_dict()
unit_by_parametre = df_merged.groupby('cdparametre')['cdunitereferencesiseeaux'].first().to_dict()

for parametre in tqdm(df_merged['cdparametre'].unique(), desc="Processing parameters"):
    if unit_by_parametre[parametre] == 'SANS OBJET':
        continue
    critere = getNomParametre(parametre)
    if critere is None:
        print(f"Parameter {parametre} not found in the API.")
        continue
    min_val = min_by_parametre[str(parametre)]
    max_val = max_by_parametre[str(parametre)]
    if max_val == 0:
        continue
    
    average_by_insee = df_merged[df_merged['cdparametre'] == parametre].groupby('inseecommuneprinc')['valtraduite'].mean()
    average_dict = average_by_insee.to_dict()
    average_dict = {str(key): value for key, value in average_dict.items()}
    
    
    plot_title = f"{critere} in {departement_code}"
    
    save_geodata(gdf, average_dict, min_val, max_val, plot_title, f'maps/{departement_code}_{critere}.png', unit_by_parametre[parametre])