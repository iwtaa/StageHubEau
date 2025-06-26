from Utilities import *


communes_potability = {}
for df in load_folder_data(os.path.join(os.getcwd(), 'data', 'clean')):
    cdparam = df['cdparametre'].iloc[0]
    for insee, group in get_by_commune(df):
        if insee not in communes_potability:
            communes_potability[insee] = True
        if group['valtraduite'].mean() > 0.0:
            communes_potability[insee] += 1.0
