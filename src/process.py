from parameter_plot import *
import pandas as pd
from tqdm import tqdm
from sklearn.linear_model import LinearRegression
import numpy as np

def main():
    files = load_data_list()
    files = files[::-1]
    for file in tqdm(files):
        pdf = load_and_prepare_data(file)
        pdf = center_reduce(pdf)
        std_devs = pdf.groupby("inseecommuneprinc")["valtraduite"].std()
        pdf["stdcommune"] = pdf["inseecommuneprinc"].map(std_devs)
        mean_vals = pdf.groupby("inseecommuneprinc")["valtraduite"].mean()
        pdf["meancommune"] = pdf["inseecommuneprinc"].map(mean_vals)
        global_mean = pdf["valtraduite"].mean()
        global_std = pdf["valtraduite"].std()
        pdf["deregionalized_valtraduite"] = (pdf["centered_reduced_val"] * global_std) + global_mean
        pdf = smooth(pdf, col='deregionalized_valtraduite', window_size=60)
        pdf = smooth(pdf, col='centered_reduced_val', window_size=60)
        file_name = file.split(".")[0]
        pdf.to_csv(f"{file_name}_processed.csv")

if __name__ == "__main__":
    main()
