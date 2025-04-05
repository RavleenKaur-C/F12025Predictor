import fastf1
import pandas as pd
import os

os.makedirs('f1_cache', exist_ok=True)
fastf1.Cache.enable_cache('f1_cache')

# Load and process lap time data for a specific race
def get_avg_laptimes(year, gp_name):
    try:
        session = fastf1.get_session(year, gp_name, 'Race')
        session.load()
        laps = session.laps

        # Drop NaN lap timeDs and convert to seconds
        laps = laps.dropna(subset=['LapTime'])
        laps['LapTimeSeconds'] = laps['LapTime'].dt.total_seconds()

        # Calculate average lap time per driver
        avg_laptimes = laps.groupby('Driver')["LapTimeSeconds"].mean().reset_index()
        avg_laptimes.columns = ['Driver', f'AvgLapTime_{year}']

        return avg_laptimes
    except Exception as e:
        print(f"Could not process {gp_name} {year}:", e)
        return pd.DataFrame()


def compute_wet_performance():
    # Get average lap times for Canada 2024 (wet) and Canada 2023 (dry)
    dry_laps = get_avg_laptimes(2023, "Canada")
    wet_laps = get_avg_laptimes(2024, "Canada")

    # Merge both datasets on driver
    merged = pd.merge(dry_laps, wet_laps, on='Driver')

    # Absolute difference and percentage change
    merged['TimeDiff'] = merged['AvgLapTime_2023'] - merged['AvgLapTime_2024']
    merged['PercentChange'] = (merged['TimeDiff'] / merged['AvgLapTime_2024']) * 100

    # Performance score - the higher, the better they are in the wet conditions relatively
    merged['WetPerformanceScore'] = 1 + merged['PercentChange']/100
    merged = merged.sort_values(by='WetPerformanceScore', ascending=False)

    # Print final result
    print("\nWet Performance Score:")
    print(merged[['Driver', 'WetPerformanceScore']])


if __name__ == "__main__":
    compute_wet_performance()