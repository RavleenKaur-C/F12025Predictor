import fastf1
import pandas as pd
import requests
from datetime import datetime
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import os
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from wet_performance_score import compute_wet_performance

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

driver_to_team = {
    "Max Verstappen": "Red Bull", "Lando Norris": "McLaren", "Oscar Piastri": "McLaren",
    "George Russell": "Mercedes", "Charles Leclerc": "Ferrari", "Lewis Hamilton": "Ferrari",
    "Carlos Sainz": "Williams", "Kimi Antonelli": "Mercedes", "Isack Hadjar": "Racing Bulls",
        "Yuki Tsunoda": "Red Bull", "Alexander Albon": "Williams", "Pierre Gasly": "Alpine",
    "Esteban Ocon": "Haas", "Fernando Alonso": "Aston Martin", "Nico Hulkenberg": "Kick Sauber",
    "Jack Doohan": "Alpine", "Oliver Bearman": "Haas", "Lance Stroll": "Aston Martin",
    "Gabriel Bortoleto": "Kick Sauber", "Liam Lawson": "Racing Bulls"}

team_points = {
    "Red Bull": 71, "Ferrari": 57, "McLaren": 151, "Mercedes": 93,
    "Aston Martin": 10, "Racing Bulls": 7, "Haas": 20, "Williams": 19,
    "Alpine": 6, "Kick Sauber": 6
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
    session = fastf1.get_session(year, gp_name, "Race")
    session.load()
    laps = session.laps[["Driver", "Sector1Time", "Sector2Time", "Sector3Time"]].dropna()
    for s in ["Sector1Time", "Sector2Time", "Sector3Time"]:
        laps[f"{s} (s)"] = laps[s].dt.total_seconds()
    agg = laps.groupby("Driver")[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].mean().reset_index()
    agg["SectorAggregate (s)"] = agg[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)
    return agg

def load_qualifying_data():
    df = pd.DataFrame({
        "Driver": [
            "Max Verstappen", "Oscar Piastri", "George Russell", "Charles Leclerc", "Kimi Antonelli",
            "Carlos Sainz", "Lewis Hamilton", "Yuki Tsunoda", "Pierre Gasly", "Lando Norris",
            "Alexander Albon", "Liam Lawson", "Fernando Alonso", "Isack Hadjar", "Oliver Bearman",
            "Lance Stroll","Jack Doohan", "Nico Hulkenberg", "Esteban Ocon", "Gabriel Bortoleto",
        ],
        "QualifyingTime (s)": [87.294, 87.304, 87.407, 87.670, 87.866, 88.164, 88.201, 88.204, 88.367,
                               87.481, 88.109, 88.191, 88.303, 88.418, 88.648, 88.645, 88.739, 88.782,
                               89.092, 89.642]
    })
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    df["Team"] = df["Driver"].map(driver_to_team)
    df["TeamPoints"] = df["Team"].map(team_points)
    return df

def prepare_features(qual_df, wet_df, sector_df, use_wet):
    merged = qual_df.copy()
    merged = pd.merge(merged, sector_df, how="left", left_on="DriverCode", right_on="Driver")
    merged.drop(columns=[col for col in ["Driver_y"] if col in merged.columns], inplace=True)
    merged.rename(columns={"Driver_x": "Driver"}, inplace=True)
    if use_wet and wet_df is not None:
        merged = pd.merge(merged, wet_df, how="left", left_on="DriverCode", right_on="Driver")
        merged["WetPerformanceScore"] = merged["WetPerformanceScore"].fillna(wet_df["WetPerformanceScore"].mean())
    else:
        merged["WetPerformanceScore"] = 0.0
    merged["SectorAggregate (s)"] = merged[["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]].sum(axis=1)

    # log transform for outliers
    merged["SectorAggregate (s)"] = merged["SectorAggregate (s)"].apply(lambda x: np.log1p(x))

    features = ["QualifyingTime (s)", "WetPerformanceScore", "SectorAggregate (s)", "TeamPoints"]
    return merged[features], merged

def apply_weights_and_scale(X_df, weights_dict):
    X_weighted = X_df.copy()
    for col, weight in weights_dict.items():
        if col in X_weighted.columns:
            X_weighted[col] *= weight

    # Now scale everything between 0 and 1
    scaler = MinMaxScaler()
    X_scaled = pd.DataFrame(scaler.fit_transform(X_weighted), columns=X_weighted.columns, index=X_weighted.index)
    return X_scaled

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
    print("\n Predicted 2025 Jeddah GP Order:")
    print(result[["Driver", "PredictedRaceTime (s)"]])
    return result

if __name__ == "__main__":
    # === Location & Date for Saudi GP ===
    lat, lon = get_coordinates_from_city("Jeddah", api_key)
    race_datetime = datetime(2025, 4, 20, 17, 0)
    use_wet_performance = is_rain_expected(api_key, lat, lon, race_datetime)

    print("use_wet_performance =", use_wet_performance)

    print("Getting Wet Performance Score...")
    wet_scores = compute_wet_performance(return_df=True) if use_wet_performance else None

    print("Loading Saudi 2025 Qualifying Data...")
    saudi_qual = load_qualifying_data()

    print("Getting sector times from Saudi 2024...")
    sector_times = get_avg_sector_times(2024, "Saudi Arabia")

    print("Preparing feature matrix...")
    X_all, merged_all = prepare_features(saudi_qual, wet_scores, sector_times, use_wet_performance)

    print("Loading Saudi 2024 race times as labels...")
    y_source = get_avg_laptimes(2024, "Saudi")
    full_merged = pd.merge(merged_all, y_source, how="left", left_on="DriverCode", right_on="Driver")
    full_merged.dropna(subset=["AvgLapTime_2024"], inplace=True)

    weights = {
        "QualifyingTime (s)": 1.0,
        "WetPerformanceScore": 1.0 if use_wet_performance else 0.0,
        "SectorAggregate (s)": 0.1,
        "TeamPoints": 0.4
    }

    print("Applying feature weights...")
    X_final = apply_weights_and_scale(
        full_merged[["QualifyingTime (s)", "WetPerformanceScore", "SectorAggregate (s)", "TeamPoints"]], weights)
    y_final = full_merged["AvgLapTime_2024"]

    print("Training model...")
    model = train_model(X_final, y_final)

    print("Predicting 2025 Saudi GP results...")
    X_predict = apply_weights_and_scale(X_all, weights)
    predict(model, X_predict, merged_all)

    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=merged_all,
        x="TeamPoints",
        y="PredictedRaceTime (s)",
        hue="QualifyingTime (s)",
        palette="viridis",
        size="QualifyingTime (s)",
        sizes=(60, 200),
        legend="brief"
    )
    plt.title("Predicted Race Time vs Team Points (Saudi Arabia 2025)")
    plt.xlabel("Team Points")
    plt.ylabel("Predicted Race Time (s)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Permutation Feature Importance
    print("Calculating feature importance...")
    result = permutation_importance(model, X_final, y_final, n_repeats=10, random_state=0)
    importance_df = pd.DataFrame({
        "Feature": X_final.columns,
        "Importance": result.importances_mean
    }).sort_values(by="Importance", ascending=False)

    plt.figure(figsize=(8, 5))
    sns.barplot(data=importance_df, x="Importance", y="Feature", palette="Blues_d")
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()
