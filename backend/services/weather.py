import requests

API_KEY = "95bb9fa86afc670c4b423b8de1d2c2b5"

def fetch_weather(lat, lon):
    url = (
        "https://api.openweathermap.org/data/2.5/weather"
        f"?lat={lat}&lon={lon}&appid={API_KEY}&units=metric"
    )
    r = requests.get(url)
    r.raise_for_status()
    d = r.json()
    return {
        "temp": d["main"]["temp"],
        "humidity": d["main"]["humidity"],
        "pressure": d["main"]["pressure"],
        "wind": d["wind"]["speed"]
    }
