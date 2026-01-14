import json

import requests

# Specify the search term
query = '"generative ai"'
year_range = 2020, 2025

# Define the API endpoint URL
url = "http://api.semanticscholar.org/graph/v1/paper/search/bulk"

# Define the query parameters
query_params = {
    "query": '"generative ai"',
    "fields": "title,url,publicationTypes,publicationDate,openAccessPdf",
    "year": f"{year_range[0]}-{year_range[1]}",
}

# Directly define the API key (Reminder: Securely handle API keys in production environments)
api_key = "your api key goes here"  # Replace with the actual API key

# Define headers with API key
headers = {"x-api-key": api_key}

# Send the API request
response = requests.get(url, params=query_params, headers=headers).json()
