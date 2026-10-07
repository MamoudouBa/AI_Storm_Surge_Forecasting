#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

import noaa_coops as nc
import pandas as pd
import numpy as np
import os
import time

#
# --- 1. Leap Year Function ---
#
def leap_year(y):
    """Checks if a given integer year is a leap year."""
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False

#
# --- 2. Configuration ---
#
print("Script starting...")

# Data ranges
year = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010',
        '2011', '2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019',
        '2020', '2021', '2022', '2023', '2024', '2025']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
       '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
       '25', '26', '27', '28', '29', '30', '31']

# Base number of days (will be copied)
number_of_day_in_a_month_base = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

# Station
ft_myers = nc.Station(8725520)

# --- Define output file name ONCE ---
output_file_name = 'training_test_ft_myers_for_water_level.csv'

# --- (Optional) Uncomment this block to start with a fresh file ---
# if os.path.exists(output_file_name):
#     print(f"Removing existing file: {output_file_name}")
#     os.remove(output_file_name)
# -----------------------------------------------------------------

#
# --- 3. Main Data Fetching Loop ---
#
# Loop from index 1 ('2003') to 19 ('2021')
for iyear in range(21, 22):
    current_year_str = year[iyear]
    current_year_int = int(current_year_str)
    
    number_of_day_in_a_month = number_of_day_in_a_month_base.copy()
    if leap_year(current_year_int):
        number_of_day_in_a_month[1] = 29  # Set Feb to 29 days

    for imonth in range(0, 12):
        current_month_str = month[imonth]
        days_in_this_month = number_of_day_in_a_month[imonth]
        
        beginDate = current_year_str + current_month_str + day[0]
        endDate = current_year_str + current_month_str + str(days_in_this_month)
        
        print(f"--- Fetching data for {current_year_str}-{current_month_str} ---")
        
        try:
            df_water_levels = ft_myers.get_data(
                begin_date=beginDate, end_date=endDate,
                product="water_level", datum="MLLW",
                units="metric", time_zone="gmt")

            df_air_temp = ft_myers.get_data(
                begin_date=beginDate, end_date=endDate,
                product="air_temperature",
                units="metric", time_zone="gmt")

            df_winds = ft_myers.get_data(
                begin_date=beginDate, end_date=endDate,
                product="wind",
                units="metric", time_zone="gmt")

            df_air_pressure = ft_myers.get_data(
                begin_date=beginDate, end_date=endDate,
                product="air_pressure",
                units="metric", time_zone="gmt")

        except ValueError as e:
            print(f"Caught an error, no data for this period: {e}")
            continue # Skip to the next month

        # -----------------------------------------------------------------
        # --- ROBUST FIX: Check for required columns BEFORE use ---
        # -----------------------------------------------------------------
        # This solves the KeyError by verifying the column exists.
        
        if df_water_levels is None or 'water_level' not in df_water_levels.columns:
            print("No 'water_level' column found. Skipping month.")
            continue
            
        if df_air_temp is None or 'air_temperature' not in df_air_temp.columns:
            print("No 'air_temperature' column found. Skipping month.")
            continue
            
        # Wind needs multiple columns
        required_wind_cols = ['wind_speed', 'wind_direction', 'wind_gusts']
        if df_winds is None or not all(col in df_winds.columns for col in required_wind_cols):
            # Fallback for 'wind_gusts' which is sometimes missing
            if df_winds is not None and 'wind_speed' in df_winds.columns and 'wind_direction' in df_winds.columns:
                 print("Warning: 'wind_gusts' column missing. Filling with NaN.")
                 df_winds['wind_gusts'] = np.nan
            else:
                print("Missing essential wind columns ('wind_speed', 'wind_direction'). Skipping month.")
                continue

        if df_air_pressure is None or 'air_pressure' not in df_air_pressure.columns:
            print("No 'air_pressure' column found. Skipping month.")
            continue
        # --- END FIX ---

        
        # 2. Drop NaNs (now safe)
        df_water_levels = df_water_levels.dropna(subset=['water_level'])
        df_air_temp = df_air_temp.dropna(subset=['air_temperature'])
        df_winds = df_winds.dropna(subset=['wind_speed', 'wind_direction']) # 'wind_gusts' can be NaN
        df_air_pressure = df_air_pressure.dropna(subset=['air_pressure'])

        # Check if they became empty after dropping NaNs
        if df_water_levels.empty or df_air_temp.empty or df_winds.empty or df_air_pressure.empty:
            print("Data became empty after dropping NaNs. Skipping month.")
            continue

        # 3. Select and rename columns (now safe)
        wl = df_water_levels[['water_level']]
        at = df_air_temp[['air_temperature']].rename(columns={'air_temperature': 'air_temp'})
        ap = df_air_pressure[['air_pressure']]
        w = df_winds[['wind_speed', 'wind_direction', 'wind_gusts']].rename(
            columns={'wind_gusts': 'wind_gust'})

        # 4. Join all dataframes on their (time) index
        df_joined = wl.join([at, ap, w], how='inner')

        if df_joined.empty:
            print("No aligned data found for this month.")
            continue

        # 5. Filter for hourly data (minute == 0)
        df_hourly = df_joined[df_joined.index.minute == 0].copy()

        if df_hourly.empty:
            print("No hourly (00-minute) data found for this month.")
            continue
            
        # 6. Create the 'timestamp' column from the index
        df_hourly['timestamp'] = df_hourly.index.strftime('%Y-%m-%d %H %M %S')
        
        # 7. Reorder columns
        final_columns = [
            'timestamp', 'air_temp', 'air_pressure', 'wind_speed',
            'wind_direction', 'wind_gust', 'water_level'
        ]
        df_final = df_hourly[final_columns]

        # 8. --- HEADER FIX: Check if file exists ---
        write_header = not os.path.exists(output_file_name)
        if write_header:
            print("File not found, writing new file with header.")
        
        # 9. Append to CSV
        df_final.to_csv(
            output_file_name,
            index=False,
            mode='a',
            header=write_header,
            float_format='%g'
        )
        
        print(f"Successfully added {len(df_final)} hourly records for {current_year_str}-{current_month_str}")
        time.sleep(0.5)

print(f"\nScript finished. Data saved to {output_file_name}")

