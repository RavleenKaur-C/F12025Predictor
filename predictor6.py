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
    "Gabriel Bortoleto": "Kick Sauber", "Liam Lawson": "Racing Bulls"
}

team_points = {
    "Red Bull": 89, "Ferrari": 78, "McLaren": 188, "Mercedes": 111,
    "Aston Martin": 10, "Racing Bulls": 8, "Haas": 20, "Williams": 25,
    "Alpine": 6, "Kick Sauber": 6
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

def is_rain_expected(api_key, lat, lon, race_date):
    url = "https://api.openweathermap.org/data/3.0/onecall"
    params = {'lat': lat, 'lon': lon, 'exclude': 'minutely,hourly,alerts', 'appid': api_key}
    response = requests.get(url, params=params).json()
    for day in response.get('daily', []):
        date = datetime.fromtimestamp(day['dt']).date()
        if date == race_date.date():
            weather = day['weather'][0]['main'].lower()
            print(f"Forecast for race day: {weather}")
            return 'rain' in weather
    return False

def get_avg_laptimes(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'Race')
    session.load()
    laps = session.laps.dropna(subset=['LapTime'])
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    avg_laps = laps.groupby('Driver')['LapTimeSeconds'].mean().reset_index()
    avg_laps.columns = ['Driver', f'AvgLapTime_{year}']
    return avg_laps

def get_clean_air_pace(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'FP1')
    session.load()
    laps = session.laps.pick_quicklaps().pick_accurate()
    if laps.empty:
        print("No clean laps found, returning empty DataFrame")
        return pd.DataFrame(columns=['Driver', 'CleanAirRacePace'])
    compound = laps['Compound'].mode()[0]
    print(f"Using compound: {compound}")
    clean_laps = laps[(laps['Compound'] == compound) & (laps['TrackStatus'] == 1)]
    clean_air_avg = clean_laps.groupby('Driver')['LapTime'].apply(lambda x: x.dt.total_seconds().mean()).reset_index()
    clean_air_avg.columns = ['Driver', 'CleanAirRacePace']
    return clean_air_avg

def load_dummy_qualifying():
    df = pd.DataFrame({
        "Driver": list(driver_mapping.keys()),
        "QualifyingTime": np.random.uniform(87, 90, len(driver_mapping))
    })
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    df["Team"] = df["Driver"].map(driver_to_team)
    df["TeamPoints"] = df["Team"].map(team_points)
    return df

# Main pipeline
if __name__ == "__main__":
    lat, lon = get_coordinates("Miami", api_key)
    race_date = datetime(2025, 5, 4, 16, 0)
    raining = is_rain_expected(api_key, lat, lon, race_date)

    wet_scores = compute_wet_performance(return_df=True) if raining else None
    quali = load_dummy_qualifying()
    clean_air = get_clean_air_pace(2024, "Miami")
    past_race_times = get_avg_laptimes(2024, "Miami")

    merged = quali.merge(clean_air, how='left', left_on='DriverCode', right_on='Driver') \
                  .drop(columns='Driver_y').rename(columns={'Driver_x': 'Driver'})

    if raining and wet_scores is not None:
        merged = merged.merge(wet_scores, how='left', left_on='DriverCode', right_on='Driver')
        merged['WetPerformanceScore'] = merged['WetPerformanceScore'].fillna(wet_scores['WetPerformanceScore'].mean())
    else:
        merged['WetPerformanceScore'] = 0.0

    # Ensure CleanAirRacePace is numeric
    if merged['CleanAirRacePace'].dtype == 'timedelta64[ns]':
        print("Converting CleanAirRacePace from timedelta to seconds")
        merged['CleanAirRacePace'] = merged['CleanAirRacePace'].dt.total_seconds()

    numeric_cols = ['QualifyingTime', 'CleanAirRacePace', 'TeamPoints', 'WetPerformanceScore']
    merged[numeric_cols] = merged[numeric_cols].fillna(merged[numeric_cols].mean())

    full_data = merged.merge(past_race_times, how='left', left_on='DriverCode', right_on='Driver')
    full_data = full_data.dropna(subset=['AvgLapTime_2024'])

    X = full_data[numeric_cols]
    y = full_data['AvgLapTime_2024']

    weights = {'QualifyingTime': 1.0, 'CleanAirRacePace': 0.8, 'TeamPoints': 0.4, 'WetPerformanceScore': 1.0 if raining else 0.0}
    for col, w in weights.items():
        X[col] *= w

    print("Checking for NaNs in X before training...")
    print(X.isna().sum())

    X = X.fillna(X.mean())

    X_scaled = pd.DataFrame(MinMaxScaler().fit_transform(X), columns=X.columns)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor().fit(X_train, y_train)
    print(f"Validation MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f} sec")

    predictions = model.predict(X_scaled)
    full_data['PredictedRaceTime'] = predictions
    result = full_data.sort_values(by='PredictedRaceTime')

    print("\n=== Predicted 2025 Miami GP Finishing Order ===")
    for i, row in enumerate(result[['Driver', 'PredictedRaceTime']].itertuples(), 1):
        print(f"{i}. {row.Driver} - {row.PredictedRaceTime:.3f} sec")

    # Optional: plot feature importance
    # plt.figure(figsize=(8, 5))
    # importance_df = pd.DataFrame({'Feature': X.columns, 'Importance': model.feature_importances_})
    # sns.barplot(data=importance_df.sort_values('Importance', ascending=False), x='Importance', y='Feature', palette='Blues_d')
    # plt.title("Feature Importance")
    # plt.tight_layout()
    # plt.show()