# test_client.py

import requests

# Define the URL for the web service
url = 'http://localhost:9696/predict'

# Define client data to be scored
client_id = 'client-xyz'
client_data = {
    "job": "student",
    "duration": 280,
    "poutcome": "failure"
}

# Send the request to the web service and get the response
response = requests.post(url, json=client_data).json()
print(response)

# Process the result and print an appropriate message
if response['subscription']:
    print(f'Sending subscription offer to {client_id}')
else:
    print(f'No subscription offer sent to {client_id}')