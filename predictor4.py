import fastf1
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from wet_performance_score import compute_wet_performance
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv("OPENWEATHER_API_KEY")

# Enable FastF1 caching
fastf1.Cache.enable_cache('f1_cache')

# Weather config with openweather api
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
                print(f"Forecast for race day: {weather}")
                return 'rain' in weather
    except Exception as e:
        print("Failed to fetch weather data:", e)
    return False

# Map full names to FastF1 driver codes
driver_mapping = {
    "Lando Norris": "NOR", "Oscar Piastri": "PIA", "Max Verstappen": "VER",
    "George Russell": "RUS", "Yuki Tsunoda": "TSU", "Alexander Albon": "ALB",
    "Charles Leclerc": "LEC", "Lewis Hamilton": "HAM", "Pierre Gasly": "GAS",
    "Carlos Sainz": "SAI", "Fernando Alonso": "ALO", "Esteban Ocon": "OCO",
    "Lance Stroll": "STR", "Logan Sargeant": "SAR", "Nico Hulkenberg": "HUL",
    "Valtteri Bottas": "BOT", "Zhou Guanyu": "ZHO", "Kevin Magnussen": "MAG",
    "Oliver Bearman": "BEA", "Jack Doohan": "DOO"
}

def get_coordinates_from_city(city_name, api_key):
    print(f"Getting coordinates for {city_name}...")
    url = f"http://api.openweathermap.org/geo/1.0/direct"
    params = {
        'q': city_name,
        'limit': 1,
        'appid': api_key
    }
    try:
        response = requests.get(url, params=params)
        data = response.json()
        if len(data) > 0:
            lat = data[0]['lat']
            lon = data[0]['lon']
            print(f"Coordinates: lat={lat}, lon={lon}")
            return lat, lon
        else:
            print("Could not find coordinates.")
            return None, None
    except Exception as e:
        print("Error getting coordinates:", e)
        return None, None


def get_avg_laptimes(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'Race')
    session.load()
    laps = session.laps.dropna(subset=['LapTime'])
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    avg_laps = laps.groupby('Driver')['LapTimeSeconds'].mean().reset_index()
    avg_laps.columns = ['Driver', f'AvgLapTime_{year}']
    return avg_laps

def get_avg_sector_times(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'Race')
    session.load()
    laps = session.laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()
    for col in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{col} (s)"] = laps[col].dt.total_seconds()
    avg_sectors = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    return avg_sectors

def load_qualifying_data():
    df = pd.DataFrame({
        "Driver": [
            "Oscar Piastri", "George Russell", "Charles Leclerc",  "Kimi Antonelli",
            "Pierre Gasly", "Lando Norris", "Max Verstappen", "Carlos Sainz",
            "Lewis Hamilton", "Yuki Tsunoda",  "Jack Doohan", "Isack Hadjar",
            "Fernando Alonso", "Esteban Ocon", "Alexander Albon", "Nico Hulkenberg",
            "Liam Lawson", "Gabriel Bortoleto", "Lance Stroll", "Oliver Bearman"
        ],
        "QualifyingTime (s)": [89.841, 90.009, 90.175, 90.213, 90.216, 90.267, 90.423, 90.680,
                               90.772, 91.303, 91.245, 91.271, 91.886, 91.594, 92.040, 92.067,
                               92.165, 92.186, 92.283, 92.373]
    })
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    return df

def prepare_features(qual_df, wet_df, sector_df, use_wet):
    merged = qual_df.copy()

    if use_wet and wet_df is not None:
        merged = pd.merge(merged, wet_df, how="left", left_on="DriverCode", right_on="Driver")
        merged.drop(columns=[col for col in ["Driver_y"] if col in merged.columns], inplace=True)
        merged.rename(columns={"Driver_x": "Driver"}, inplace=True)
        avg_score = wet_df["WetPerformanceScore"].mean()
        merged["WetPerformanceScore"] = merged["WetPerformanceScore"].fillna(avg_score)
    else:
        merged["WetPerformanceScore"] = 0.0

    merged = pd.merge(merged, sector_df, how="left", left_on="DriverCode", right_on="Driver")
    merged.drop(columns=[col for col in ["Driver_y"] if col in merged.columns], inplace=True)
    merged.rename(columns={"Driver_x": "Driver"}, inplace=True)

    for sector in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
        mean_val = merged[sector].mean()
        merged[sector] = merged[sector].fillna(mean_val)

    X = merged[["QualifyingTime (s)", "WetPerformanceScore", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    return X, merged

def apply_manual_weights(X_df, weights_dict):
    X_weighted = X_df.copy()
    for col, weight in weights_dict.items():
        if col in X_weighted.columns:
            X_weighted[col] *= weight
    return X_weighted

def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    print(f"MAE: {mean_absolute_error(y_test, model.predict(X_test)):.3f} sec")
    return model

def predict(model, X, merged_df):
    preds = model.predict(X)
    merged_df["PredictedRaceTime (s)"] = preds
    result = merged_df.sort_values(by="PredictedRaceTime (s)")
    print("\n Predicted 2025 Bahrain GP Order:")
    print(result[["Driver", "PredictedRaceTime (s)"]])
    return result

if __name__ == "__main__":
    # === Location & Date for Bahrain GP ===
    api_key = 'YOUR_API_KEY'  # 🔑 Replace with your OpenWeatherMap API key
    lat, lon = get_coordinates_from_city("Bahrain", api_key)
    race_datetime = datetime(2025, 4, 13, 18, 0)  # UTC time for Sunday race start

    use_wet_performance = is_rain_expected(api_key, lat, lon, race_datetime)

    print("use_wet_performance =", use_wet_performance)

    print("Getting Wet Performance Score...")
    wet_scores = compute_wet_performance(return_df=True) if use_wet_performance else None

    print("Loading Bahrain 2025 Qualifying Data...")
    bahrain_qual = load_qualifying_data()

    print("Getting sector times from Bahrain 2024...")
    sector_times = get_avg_sector_times(2024, "Bahrain")

    print("Preparing feature matrix...")
    X_all, merged_all = prepare_features(bahrain_qual, wet_scores, sector_times, use_wet_performance)

    print("Loading Bahrain 2024 race times as labels...")
    y_source = get_avg_laptimes(2024, "Bahrain")
    full_merged = pd.merge(merged_all, y_source, how="left", left_on="DriverCode", right_on="Driver")
    full_merged.dropna(subset=["AvgLapTime_2024"], inplace=True)

    weights = {
        "QualifyingTime (s)": 1.0,
        "WetPerformanceScore": 1.0 if use_wet_performance else 0.0,
        "Sector1Time (s)": 0.3,
        "Sector2Time (s)": 0.3,
        "Sector3Time (s)": 0.3,
    }

    print("Applying feature weights...")
    X_final = apply_manual_weights(
        full_merged[["QualifyingTime (s)", "WetPerformanceScore", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]],
        weights
    )
    y_final = full_merged["AvgLapTime_2024"]

    print("Training model...")
    model = train_model(X_final, y_final)

    print("Predicting 2025 Bahrain GP results...")
    X_predict = apply_manual_weights(X_all, weights)
    predict(model, X_predict, merged_all)
