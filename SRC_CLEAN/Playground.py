import csv
import pandas as pd
import numpy as np
from tqdm import tqdm

# Specify the path to your CSV file
csv_file_path = 'data/PAR_20250523_SANDRE.csv'

# Read the CSV file into a Pandas DataFrame
df = pd.read_csv(csv_file_path, delimiter=';')

'''
# create a txt file to write the results
with open('temp/columns_stats.txt', 'w') as f:
    # Iterate over each column in the DataFrame
    for column in df.columns:
        # Calculate the number of NaN values in the column
        nan_count = df[column].isna().sum()
        
        # Calculate the number of unique values in the column
        unique_count = df[column].nunique()
        
        if nan_count > 2000:
            continue
        # Write the results to the txt file
        f.write(f'{column} - nan:{nan_count} - unique:{unique_count}\n')
'''

columns_to_export = [
    "CdParametre",
    "NomParametre",
    "StParametre",
    "DfParametre",
    "NatParametre",
    "TypeParametre",
    "ParametreCalcule",
    "LbCourtParametre",
    "ParametreChimique",
    "CdGroupeParametres1",
    "NomGroupeParametres1",
    "DateModificationPar1",
    "TypeGenealogiePar1",
    "ComGenealogiePar1",
    "CdUniteMesure1",
    "LbUniteMesure1",
    "SymUniteMesure1",
    "CdNatureFractionAnalysee1",
    "LbNatureFractionAnalysee1",
]

# Export unique values to individual files
for column in tqdm(columns_to_export, desc="Exporting columns"):
    unique_values = df[column].dropna().unique()
    
    # Create a filename based on the column name
    filename = f"temp/{column}_unique_values.txt"
    
    # Write the unique values to the file
    with open(filename, 'w', encoding='utf-8') as f:
        for value in unique_values:
            f.write(str(value) + '\n')
