import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from wet_performance_score import compute_wet_performance

# Enable FastF1 caching
fastf1.Cache.enable_cache('f1_cache')

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
        "Driver": ["Max Verstappen", "Lando Norris", "Oscar Piastri", "Charles Leclerc",
                   "George Russell", "Kimi Antonelli", "Isack Hadjar", "Lewis Hamilton",
                   "Alexander Albon", "Oliver Bearman", "Pierre Gasly", "Fernando Alonso",
                   "Liam Lawson", "Yuki Tsunoda", "Carlos Sainz", "Nico Hulkenberg",
                   "Gabriel Bortoleto", "Esteban Ocon", "Jack Doohan", "Lance Stroll"],
        "QualifyingTime (s)": [86.983, 86.995, 87.027, 87.299, 87.318, 87.555, 87.610, 87.615,
                               87.867, 87.822, 87.836, 87.987, 87.906, 88.000, 87.836, 88.570,
                               88.622, 88.696, 88.877, 89.271],
    })
    df["DriverCode"] = df["Driver"].map(driver_mapping)
    return df

def prepare_features(qual_df, wet_df, sector_df):
    merged = pd.merge(qual_df, wet_df, how="left", left_on="DriverCode", right_on="Driver")
    merged.drop(columns=[col for col in ["Driver_y"] if col in merged.columns], inplace=True)
    merged.rename(columns={"Driver_x": "Driver"}, inplace=True)
    avg_score = wet_df["WetPerformanceScore"].mean()
    merged["WetPerformanceScore"] = merged["WetPerformanceScore"].fillna(avg_score)

    # Merge sector times
    merged = pd.merge(merged, sector_df, how="left", left_on="DriverCode", right_on="Driver")
    merged.drop(columns=[col for col in ["Driver_y"] if col in merged.columns], inplace=True)
    merged.rename(columns={"Driver_x": "Driver"}, inplace=True)

    for sector in ["Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]:
        mean_val = merged[sector].mean()
        merged[sector] = merged[sector].fillna(mean_val)

    X = merged[["QualifyingTime (s)", "WetPerformanceScore", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]]
    return X, merged

#Weighting Logic
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
    print("\nPredicted 2025 Japan GP Order (Weighted Features):")
    print(result[["Driver", "PredictedRaceTime (s)"]])
    return result

if __name__ == "__main__":
    print("Computing Wet Performance Score...")
    wet_scores = compute_wet_performance(return_df=True)

    print("Loading Japan 2025 Qualifying Data...")
    japan_qual = load_qualifying_data()

    print("Getting sector times from Japan 2024...")
    sector_times = get_avg_sector_times(2024, "Japan")

    print("Preparing feature matrix...")
    X_all, merged_all = prepare_features(japan_qual, wet_scores, sector_times)

    print("Loading Japan 2024 race times as labels...")
    y_source = get_avg_laptimes(2024, "Japan")
    full_merged = pd.merge(merged_all, y_source, how="left", left_on="DriverCode", right_on="Driver")
    full_merged.dropna(subset=["AvgLapTime_2024"], inplace=True)

    # Feature Weights
    weights = {
        "QualifyingTime (s)": 1.2,
        "WetPerformanceScore": 1.0,
        "Sector1Time (s)": 0.2,
        "Sector2Time (s)": 0.2,
        "Sector3Time (s)": 0.2,
    }

    print("Applying feature weights...")
    X_final = apply_manual_weights(
        full_merged[["QualifyingTime (s)", "WetPerformanceScore", "Sector1Time (s)", "Sector2Time (s)", "Sector3Time (s)"]],
        weights
    )
    y_final = full_merged["AvgLapTime_2024"]

    print("Training model...")
    model = train_model(X_final, y_final)

    print("Predicting 2025 Japan GP results...")
    X_predict = apply_manual_weights(X_all, weights)
    predict(model, X_predict, merged_all)
