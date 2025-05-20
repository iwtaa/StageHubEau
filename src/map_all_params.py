import pandas as pd
import json

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
