import requests

def fetch_no2(city):
    url = f"https://api.openaq.org/v2/latest?city={city}&parameter=no2"
    r = requests.get(url)
    if r.status_code != 200:
        return None
    data = r.json()["results"]
    if not data:
        return None
    return data[0]["measurements"][0]["value"]
