import requests

BASE = "http://127.0.0.1:5000/"

response = requests.put(BASE + "text", {'text': "buna ziua domnule presedinte"})
print(response.json())
