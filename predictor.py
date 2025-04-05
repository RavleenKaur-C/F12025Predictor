import fastf1
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

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

# Get average lap times for a given race
def get_avg_laptimes(year, gp_name):
    session = fastf1.get_session(year, gp_name, 'Race')
    session.load()
    laps = session.laps.dropna(subset=['LapTime'])
    laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()
    avg_laps = laps.groupby('Driver')['LapTimeSeconds'].mean().reset_index()
    avg_laps.columns = ['Driver', f'AvgLapTime_{year}']
    return avg_laps

# Compute wet performance score from Canada GP (2023 dry vs 2024 wet)
def compute_wet_performance():
    dry_laps = get_avg_laptimes(2023, 'Canada')
    wet_laps = get_avg_laptimes(2024, 'Canada')
    merged = pd.merge(dry_laps, wet_laps, on='Driver')
    merged['WetPerformanceScore'] = merged['AvgLapTime_2023'] / merged['AvgLapTime_2024']
    return merged[['Driver', 'WetPerformanceScore']]

# Japan 2025 qualifying data (manually input)
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

# Combine qualifying with wet score features
def prepare_features(qual_df, wet_df):
    merged = pd.merge(qual_df, wet_df, how="left", left_on="DriverCode", right_on="Driver")
    merged = merged.drop(columns=["Driver_y"]).rename(columns={"Driver_x": "Driver"})
    avg_score = wet_df["WetPerformanceScore"].mean()
    merged["WetPerformanceScore"] = merged["WetPerformanceScore"].fillna(avg_score)
    X = merged[["QualifyingTime (s)", "WetPerformanceScore"]]
    return X, merged

# Train model on past race results
def train_model(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = GradientBoostingRegressor()
    model.fit(X_train, y_train)
    print(f"Mean Absolute Error: {mean_absolute_error(y_test, model.predict(X_test)):.3f} sec")
    return model

# Predict race outcome
def predict(model, X, merged_df):
    preds = model.predict(X)
    merged_df["PredictedRaceTime (s)"] = preds
    result = merged_df.sort_values(by="PredictedRaceTime (s)")
    print("\nPredicted 2025 Japan GP Order (Wet Conditions):")
    print(result[["Driver", "QualifyingTime (s)", "WetPerformanceScore", "PredictedRaceTime (s)"]])
    return result

if __name__ == "__main__":
    print("Computing Wet Performance Score...")
    wet_scores = compute_wet_performance()

    print("Loading Japan 2025 Qualifying Data...")
    japan_qual = load_qualifying_data()

    print("Merging Feature Set...")
    X_all, merged_all = prepare_features(japan_qual, wet_scores)

    print("Loading Japan 2024 Race Times as Labels...")
    y_source = get_avg_laptimes(2024, "Japan")
    full_merged = pd.merge(merged_all, y_source, how="left", left_on="DriverCode", right_on="Driver")
    full_merged = full_merged.dropna(subset=["AvgLapTime_2024"])

    # Final features & labels
    X_final = full_merged[["QualifyingTime (s)", "WetPerformanceScore"]]
    y_final = full_merged["AvgLapTime_2024"]

    print("Training Model...")
    model = train_model(X_final, y_final)

    print("Predicting 2025 Japan GP Results...")
    predict(model, X_all, merged_all)