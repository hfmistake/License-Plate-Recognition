import requests


def plate_post_request(data):
    url = "http://127.0.0.1:8000/api/predictions"
    requests.post(url, json=data)


