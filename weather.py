import requests

def is_rain_expected(api_key, lat, lon, race_datetime):
    print("Checking weather forecast...")
    url = f"https://api.openweathermap.org/data/3.0/onecall"
    params = {
        'lat': lat,
        'lon': lon,
        'exclude': 'minutely,hourly,alerts',
        'appid': api_key
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        for day in data.get('daily', []):
            date = datetime.fromtimestamp(day['dt'])
            if date.date() == race_datetime.date():
                weather = day['weather'][0]['main'].lower()
                rain = 'rain' in weather
                pop = day.get('pop', 0)  # Probability of precipitation
                temp = day['temp'].get('day', 20.0)
                return {'rain_probability': pop, 'temperature': temp}
    except Exception as e:
        print("Failed to fetch weather data:", e)
    return {'rain_probability': 0.0, 'temperature': 20.0}  # default fallback
