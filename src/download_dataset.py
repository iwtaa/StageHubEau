links = {
    '2025': 'https://www.data.gouv.fr/fr/datasets/r/6994a9f1-3f4b-4e15-a4dc-0e358a6aac13',
    '2024': 'https://www.data.gouv.fr/fr/datasets/r/c0350599-a041-4724-9942-ad4c2ba9a7b3',
    '2023': 'https://www.data.gouv.fr/fr/datasets/r/96452cf0-329a-4908-8adb-8f061adcca4c',
    '2022': 'https://www.data.gouv.fr/fr/datasets/r/77d3151a-739e-4aab-8c34-7a15d7fea55d',
    '2021': 'https://www.data.gouv.fr/fr/datasets/r/3c5ebbd9-f6b5-4837-a194-12bfeda7f38e',
    '2020': 'https://www.data.gouv.fr/fr/datasets/r/1913d0d6-d650-409d-a19e-b7c7f09e09a0',
    '2019': 'https://www.data.gouv.fr/fr/datasets/r/a6f74cfd-b4f7-44fb-8772-7884775b35e1',
    '2018': 'https://www.data.gouv.fr/fr/datasets/r/e7514726-19ec-47dc-bcc3-a59c9bfa5f7f',
    '2017': 'https://www.data.gouv.fr/fr/datasets/r/d1d5d76a-cd4a-46cc-ae7e-405424dadf7b',
    '2016': 'https://www.data.gouv.fr/fr/datasets/r/0c83108b-f87b-470d-8980-6207ac93f4eb'
}

sandre_url = 'https://api.sandre.eaufrance.fr/referentiels/v1/par.csv?outputSchema=SANDREv4'

import requests
import os
import zipfile

def main():
    years_input = input("Enter years to download [2016-2025] (comma-separated, leave empty for all): ").strip()
    if years_input:
        selected_years = [year.strip() for year in years_input.split(',') if year.strip() in links]
        if not selected_years:
            print("No valid years entered. Exiting.")
            return
        filtered_links = {year: links[year] for year in selected_years}
    else:
        filtered_links = links

    os.makedirs('data/raw', exist_ok=True)

    response = requests.get(sandre_url)
    if response.status_code == 200:
        with open('data/raw/PAR_SANDRE.csv', 'wb') as file:
            file.write(response.content)
        print("Downloaded and saved PAR_SANDRE.csv")
    else:
        print(f"Failed to download SANDRE data: {response.status_code}")

    for year, url in links.items():
        if year not in filtered_links:
            continue
        response = requests.get(url)
        if response.status_code == 200:
            zip_path = f'data/raw/dis_{year}_dept.zip'
            with open(zip_path, 'wb') as file:
                file.write(response.content)
            print(f"Downloaded data for {year}")

            # Unzip the file
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(f'data/raw/')
            print(f"Unzipped data for {year}")
        else:
            print(f"Failed to download data for {year}: {response.status_code}")

if __name__ == "__main__":
    main()