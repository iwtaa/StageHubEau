import pprint
import requests
import json

# Define the API endpoint URL
hubeau_api_url = "hubeau.eaufrance.fr/api"

nappes_api_url = "/v1/qualite_nappes/analyses"
nappesStations_api_url = "/v1/qualite_nappes/stations"

eauPotableCommunes_api_url = "/v1/qualite_eau_potable/communes_udi"
eauPotable_api_url = "/v1/qualite_eau_potable/resultats_dis"

riviere_api_url = "/v2/qualite_rivieres/analyse_pc"
riviereEnv_api_url = "/v2/qualite_rivieres/condition_environnementale_pc"
riviereOp_api_url = "/v2/qualite_rivieres/operation_pc"
riviereStations_api_url = "/v2/qualite_rivieres/station_pc"

sandre_api_url = "https://api.sandre.eaufrance.fr/referentiels/v1/"

referenciel_api_url = "par/"

def getNomParametre(code):
    """
    Function to get the parameter ID from the SANDRE API.
    :param nom: The name of the parameter to search for.
    :return: The ID of the parameter if found, None otherwise.
    """
    # Construct the full URL for the API request
    api_url = f"{sandre_api_url}{referenciel_api_url}{code}.json?outputSchema=SANDREv4"

    # Define the headers (e.g., for content type)
    headers = {
        "Content-Type": "application/json"
    }
    data = getAPIdata(api_url)
    return data['REFERENTIELS']['Referentiel']['Parametre'][0]['NomParametre'])
    

def getAPIdata(api_url):
    """
    Function to get data from an API.
    :param api_url: The URL of the API endpoint.
    :return: The JSON response from the API if the request was successful, None otherwise.
    """
    # Define the headers (e.g., for content type)
    headers = {
        "Content-Type": "application/json"
    }

    try:
        # Make the GET request
        response = requests.get(api_url, headers=headers)

        # Raise an exception for bad status codes
        response.raise_for_status()

        # Process the response
        if response.status_code == 200:
            #print("Request was successful")
            return response.json()  # Assuming the response is in JSON format
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None


getNomParametre("1062")