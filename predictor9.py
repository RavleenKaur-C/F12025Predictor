import fastf1
import pandas as pd
import numpy as np
import requests
import os
from datetime import datetime
from dotenv import load_dotenv
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns
from wet_performance_score import compute_wet_performance
from weather import is_rain_expected

# Setup
load_dotenv()
api_key = os.getenv("OPENWEATHER_API_KEY")
os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# Driver/team mappings
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER",
    "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI", "Fernando Alonso": "ALO", "Esteban Ocon": "OCO",
    "Lance Stroll": "STR", "Logan Sargeant": "SAR", "Nico Hulkenberg": "HUL",
    "Valtteri Bottas": "BOT", "Zhou Guanyu": "ZHO", "Kevin Magnussen": "MAG",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO"
}

driver_to_team = {
    "Max Verstappen": "Red Bull", "Lando Norris": "McLaren", "Oscar Piastri": "McLaren",
    "George Russell": "Mercedes", "Charles Leclerc": "Ferrari", "Lewis Hamilton": "Ferrari",
    "Carlos Sainz": "Williams", "Kimi Antonelli": "Mercedes", "Isack Hadjar": "Racing Bulls",
    "Yuki Tsunoda": "Red Bull", "Alexander Albon": "Williams", "Pierre Gasly": "Alpine",
    "Esteban Ocon": "Haas", "Fernando Alonso": "Aston Martin", "Nico Hulkenberg": "Kick Sauber",
    "Jack Doohan": "Alpine", "Oliver Bearman": "Haas", "Lance Stroll": "Aston Martin",
    "Gabriel Bortoleto": "Kick Sauber", "Liam Lawson": "Racing Bulls"}

team_points = {
    "McLaren": 319, "Mercedes": 147, "Red Bull": 143, "Williams": 54, "Ferrari": 142,
    "Haas": 26, "Aston Martin": 14, "Kick Sauber": 6, "Racing Bulls": 22, "Alpine": 7
}

# Helper functions
def get_coordinates(city, api_key):
    url = "http://api.openweathermap.org/geo/1.0/direct"
    params = {'q': city, 'limit': 1, 'appid': api_key}
    response = requests.get(url, params=params).json()
    if response:
        lat, lon = response[0]['lat'], response[0]['lon']
        print(f"Coordinates for {city}: {lat}, {lon}")
        return lat, lon
    print("City not found!")
    return None, None

def get_clean_air_pace(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'FP1')
    session.load()
    laps = session.laps.pick_quicklaps().pick_accurate()
    if laps.empty:
        return pd.DataFrame(columns=['Driver', 'CleanAirRacePace'])
    compound = laps['Compound'].mode()[0]
    clean_laps = laps[(laps['Compound'] == compound) & (laps['TrackStatus'] == 1)]
    clean_air_avg = clean_laps.groupby('Driver')['LapTime'].apply(lambda x: x.dt.total_seconds().mean()).reset_index()
    clean_air_avg.columns = ['Driver', 'CleanAirRacePace']
    return clean_air_avg

def load_qualifying():
    qualifying_data = [
        ("Oscar Piastri", 71.546),
        ("Lando Norris", 71.755),
        ("Max Verstappen", 71.848),
        ("George Russell", 71.507),
        ("Lewis Hamilton", 72.045),
        ("Kimi Antonelli", 72.111),
        ("Charles Leclerc", 72.131),
        ("Pierre Gasly", 72.199),
        ("Isack Hadjar", 72.252),
        ("Fernando Alonso", 72.284),
        ("Alexander Albon", 71.641),
        ("Gabriel Bortoleto", 72.756),
        ("Liam Lawson", 72.763),
        ("Lance Stroll", 73.058),
        ("Oliver Bearman", 73.315),
        ("Nico Hulkenberg", 73.190),
        ("Esteban Ocon", 73.201),
        ("Carlos Sainz", 73.203),
        ("Franco Colapinto", 73.334),
        ("Yuki Tsunoda", 73.385)
    ]

    df = pd.DataFrame(qualifying_data, columns=["Driver", "QualifyingTime"])
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    df["Team"] = df["Driver"].map(driver_to_team)
    df["TeamPoints"] = df["Team"].map(team_points)
    return df


# Main pipeline
if __name__ == "__main__":
    lat, lon = get_coordinates("Spain", api_key)
    race_date = datetime(2025, 6, 1, 15, 0)
    weather = is_rain_expected(api_key, lat, lon, race_date)
    rain_prob = weather['rain_probability']
    temperature = weather['temperature']

    wet_scores = compute_wet_performance(return_df=True) if rain_prob >= 0.75 else None
    quali = load_qualifying()
    clean_air = get_clean_air_pace(2024, "Spain")
    race_session = fastf1.get_session(2024, "Spain", 'Race')
    race_session.load()
    past_race_times = race_session.laps
    past_race_times = past_race_times.dropna(subset=['LapTime'])
    past_race_times = past_race_times.copy()
    past_race_times['LapTimeSeconds'] = past_race_times['LapTime'].dt.total_seconds()
    avg_race = past_race_times.groupby('Driver')['LapTimeSeconds'].mean().reset_index()
    avg_race.columns = ['Driver', 'AvgLapTime_2024']

    merged = quali.merge(clean_air, how='left', left_on='DriverCode', right_on='Driver', suffixes=('', '_CleanAir'))
    merged.drop(columns=['Driver_CleanAir'], inplace=True)
    if rain_prob >= 0.75 and wet_scores is not None:
        merged = merged.merge(wet_scores, how='left', left_on='DriverCode', right_on='Driver')
        merged['WetPerformanceScore'] = merged['WetPerformanceScore'].fillna(wet_scores['WetPerformanceScore'].mean())
    else:
        merged['WetPerformanceScore'] = 0.0

    merged['RainProbability'] = rain_prob
    merged['Temperature'] = temperature

    merged = merged.merge(avg_race, how='left', left_on='DriverCode', right_on='Driver')
    merged = merged.dropna(subset=['AvgLapTime_2024'])

    feature_cols = ['QualifyingTime', 'CleanAirRacePace', 'TeamPoints', 'WetPerformanceScore', 'RainProbability', 'Temperature']
    merged[feature_cols] = merged[feature_cols].apply(pd.to_numeric, errors='coerce').fillna(merged[feature_cols].mean())

    X = merged[feature_cols]
    y = merged['AvgLapTime_2024']

    weights = {'QualifyingTime': 1.0, 'CleanAirRacePace': 0.8, 'TeamPoints': 0.4, 'WetPerformanceScore': 1.0 if rain_prob >= 0.75 else 0.0}
    for col, w in weights.items():
        X.loc[:, col] *= w

    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor().fit(X_train, y_train)
    predictions = model.predict(X_scaled)
    merged['PredictedRaceTime'] = predictions
    merged['Driver'] = quali['Driver']

    # Sort and display only the podium (top 3 finishers)
    result = merged.sort_values(by='PredictedRaceTime')

    print("\n Predicted Spain 2025 Podium")
    for i, row in enumerate(result[['Driver']].head(3).itertuples(index=False), 1):
        print(f"{i}. {row.Driver}")

    # Keep MAE for model evaluation
    mae = mean_absolute_error(y_test, model.predict(X_test))
    print(f"\nModel Error (MAE): {mae:.2f} seconds")
