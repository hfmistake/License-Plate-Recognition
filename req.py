import requests


def plate_post_request(data):
    url = "http://127.0.0.1:8000/api/predictions"
    response = requests.post(url, json=data)
    if response.status_code == 201:
        print("Request sent successfully")
        return response.json()
    else:
        print("Error sending request")
        print(response.status_code)
        print(response.text)


