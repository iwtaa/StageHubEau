from parameter_plot import *
import pandas as pd
from tqdm import tqdm


def main():
    files = load_data_list()
    print(f"Found {len(files)} files to process.")
    for file in tqdm(files):
        pdf = load_and_prepare_data(file)
        pdf = center_reduce(pdf)
        std_devs = pdf.groupby("inseecommuneprinc")["valtraduite"].std()
        pdf["stdcommune"] = pdf.apply(lambda row: row["valtraduite"] + std_devs[row["inseecommuneprinc"]], axis=1)
        mean_vals = pdf.groupby("inseecommuneprinc")["valtraduite"].mean()
        pdf["meancommune"] = pdf.apply(lambda row: row["valtraduite"] + mean_vals[row["inseecommuneprinc"]], axis=1)
        file_name = file.split(".")[0]
        pdf.to_csv(f"{file_name}_processed.csv")

if __name__ == "__main__":
    main()