import pprint
import requests
import json

# Define the API endpoint URL
hubeau_api_url = "https://hubeau.eaufrance.fr/api"

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
        if response.status_code == 200 or response.status_code == 206:
            #print("Request was successful")
            return response.json()  # Assuming the response is in JSON format
        else:
            print(f"Request failed with status code {response.status_code}")
            return None

    except requests.exceptions.RequestException as e:
        print(f"An error occurred: {e}")
        return None

def getNomParametre(code):
    """
    Function to get the parameter ID from the SANDRE API.
    :param nom: The name of the parameter to search for.
    :return: The ID of the parameter if found, None otherwise.
    """
    # Construct the full URL for the API request
    api_url = f"{sandre_api_url}{referenciel_api_url}{code}.json?outputSchema=SANDREv4"
    data = getAPIdata(api_url)
    if data is None:
        return None
    if 'Parametre' not in data['REFERENTIELS']['Referentiel']:
        print(f"Parameter {code} not found in the response.")
        return None
    return data['REFERENTIELS']['Referentiel']['Parametre'][0]['NomParametre']


#Format dates "YYYY-MM-DD hh:mm:ss"
def getMeasureDepartment(departement, parameter=[], size=None, date_max_prelevement=None, date_min_prelevement=None):
    api_url = f"{hubeau_api_url}{eauPotable_api_url}?code_departement={departement}"
    if len(parameter) != 0:
        params_str = ",".join(map(str, parameter))
        api_url += f"&code_parametre={params_str}"
    if size:
        api_url += f"&size={size}"
    if date_max_prelevement:
        api_url += f"&date_max_prelevement={date_max_prelevement}"
    if date_min_prelevement:
        api_url += f"&date_min_prelevement={date_min_prelevement}"
    print(api_url)
    data = getAPIdata(api_url)

    return data

def getMeasureCommune(commune, parameters=[], size=None, date_max_prelevement=None, date_min_prelevement=None):
    if not parameters:
        return None
    params_str = ",".join(map(str, parameters))
    api_url = f"{hubeau_api_url}{eauPotable_api_url}?code_commune={commune}&code_parametre={params_str}"
    if size:
        api_url += f"&size={size}"
    if date_max_prelevement:
        api_url += f"&date_max_prelevement={date_max_prelevement}"
    if date_min_prelevement:
        api_url += f"&date_min_prelevement={date_min_prelevement}"
    #print(api_url)
    data = getAPIdata(api_url)
    return data

def getDepartementCommunes(departement):
    api_url = f'https://geo.api.gouv.fr/departements/{departement}/communes?'
    data = getAPIdata(api_url)
    return data


def getNumericalMeasureDepartment(departement, parameter=None, size=None, date_max_prelevement=None, date_min_prelevement=None):
    data = getMeasureDepartment(departement, parameter, size, date_max_prelevement, date_min_prelevement)
    if data is None:
        return None
    # Extract the relevant data from the response


    return data