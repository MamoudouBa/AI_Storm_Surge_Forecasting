#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
from py_noaa import coops
#this code is provided by noaa-coops
#https://api.tidesandcurrents.noaa.gov/api/prod/
#
from datetime import datetime
import noaa_coops as nc
import pandas as pd
import numpy as np
from noaa_coops import Station
import noaa_coops as nc
import zarr
import os

# 1. IMPORT THE ERROR CLASS
from noaa_coops.station import COOPSAPIError

#
# leap year function
def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False

year = ['2002', '2003', '2004', '2005', '2006', '2007', '2008', '2009', '2010', \
        '2011','2012', '2013', '2014', '2015', '2016', '2017', '2018', '2019', \
        '2020', '2021', '2022', '2023', '2024', '2025', '2026']
#year = ['2018']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
         '25', '26','27', '28', '29', '30', '31']
number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
ft_myers  = nc.Station(8725520)
#old_port_tampa  = nc.Station(8726607)
#clear_water_beatch  = nc.Station(8726724)
#st_petersburg  = nc.Station(8726520)
df_output = pd.DataFrame()
number_times = 0
#output_file_name = "test_set_2022_ft_myers_for_water_level.csv"
output_file_name = "training_set_2002_2025_ft_myers_for_water_level.csv"
if os.path.exists(output_file_name):
#   cmd = "rm test_set_2010_2021_ft_myers_for_water_level.csv"
    cmd = "rm training_set_2010_2021_ft_myers_for_water_level.csv"
    os.system(cmd)
for iyear in range(21,22):

      # 2. FIX LEAP YEAR LOGIC
      # Check the actual year and set Feb days *before* the month loop
      current_year_int = int(year[iyear])
      if leap_year(current_year_int):
          number_of_day_in_a_month[1] = 29 # Set February to 29
      else:
          number_of_day_in_a_month[1] = 28 # Set February to 28

      for imonth in range (0,12):

            beginDate = year[iyear] + month[imonth] + day[0]
            endDate = year[iyear] + month[imonth] + str(number_of_day_in_a_month[imonth])
            print(beginDate,endDate)
            #
            try:
                df_water_levels = ft_myers.get_data(
                    begin_date=beginDate,
                    end_date=endDate,
                    product="water_level",
                    datum="MLLW",
                    units="metric",
                    time_zone="gmt")
                df_air_temp = ft_myers.get_data(
                    begin_date=beginDate,
                    end_date=endDate,
                    product="air_temperature",
                    units="metric",
                    time_zone="gmt")
                df_winds = ft_myers.get_data(
                    begin_date=beginDate,
                    end_date=endDate,
                    product="wind",
                    units="metric",
                    time_zone="gmt")
                df_air_pressure = ft_myers.get_data(
                    begin_date=beginDate,
                    end_date=endDate,
                    product="air_pressure",
                    units="metric",
                    time_zone="gmt")

            # 3. FIX EXCEPT BLOCK
            # Catch both ValueError AND the specific COOPSAPIError
            except (ValueError, COOPSAPIError) as e:
                print(f"Caught an error: {e}, some obs not available in this period. Skipping.")
                continue

            #print(month[imonth],len(df_winds.values), len(df_water_levels))
            df_air_pressure = df_air_pressure.dropna()
            df_winds = df_winds.dropna()
            df_air_temp = df_air_temp.dropna()
            df_water_levels = df_water_levels.dropna()
   #air pressure: [pressure obs, 'string flag']
   #surface wind: [wind speed, wind direction in degrees, wind direction in words, gust speed, string flag]
   #air temperature: [temperature ons, string flag]
   #water level: [water level obs, sigma, qc flag, string flag]
   #print(df_air_pressure.values[0])
   #print(df_winds.values[0])
   #print(df_air_temp.values[0])
   #print(df_water_levels.values[0])
            df = pd.DataFrame()
            for i in range(0,len(df_water_levels)):
     # Parse the time series index to string index
                string_index = df_water_levels.index.strftime('%Y-%m-%d %H %M %S')
                string_wind_index = df_winds.index.strftime('%Y-%m-%d %H %M %S')
                string_air_temp_index = df_air_temp.index.strftime('%Y-%m-%d %H %M %S')
                string_air_pressure_index = df_air_pressure.index.strftime('%Y-%m-%d %H %M %S')
                #print('string_air_pressure_index, string_index ', string_air_pressure_index, string_index)
                if i<= len(string_wind_index) - 1 and string_wind_index[i] == string_index[i]:
                        w_index = i
                else:
                        continue
                if i<= len(string_air_temp_index) - 1 and string_air_temp_index[i] == string_index[i]:
                        t_index = i
                else:
                        continue
                if i<= len(string_air_pressure_index) - 1 and string_air_pressure_index[i] == string_index[i]:
                        p_index = i
                else:
                        continue
                #print('string_wind_index ', w_index)
                #print(string_air_temp_index[i])
                # Check if date of meteorological observation matches that of water level observation
                #if string_index[i] != w_index and string_index[i] != t_index and string_index[i] != p_index: continue
                # Extract hour and minute
                #print(i, len(df_air_temp), ' len of string_index ',len(string_index))
                datetime_split = string_index[i].split(' ')
                #datetime_split = string_index[i].split(' ')
                hour = datetime_split[1][0:2]
                minute = datetime_split[2][0:2]
                mn = int(minute)
                #print(i, t_index, p_index, w_index,string_index[i])
                #print(w_index[i], t_index[i], p_index[i], string_index[i])
                #print(hour,len(df_air_temp),len(df_winds), len(df_air_pressure), len(df_water_levels))
                if  mn == 0:
                      #If using all scalar values, you must pass an index
                      #reference: https://stackoverflow.com/questions/17839973
                      #/constructing-dataframe-from-values-in-variables-yields-valueerror-if-using-all
                      #
                      # df = pd.DataFrame({'timestamp':[date],['air_temp':[temp obs], 'air_pressure':[bressure obs],
                      #                      'wind_speed':[wind speed], 'wind_direction':[wind direction obs],
                      #                      'wind_gust':[gust speed], 'water_level':[water level obs]]}
                      #
                        #print(i,string_index[i], len(string_index),w_index, t_index, p_index)
                        temp_df = pd.DataFrame({'timestamp':string_index[i],'air_temp':[df_air_temp.values[t_index][0]],
                           'air_pressure':[df_air_pressure.values[p_index][0]],'wind_speed':[df_winds.values[w_index][0]],
                           'wind_direction':[df_winds.values[w_index][1]],
                           'wind_gust':[df_winds.values[w_index][3]],'water_level': [df_water_levels.values[i][0]]})
                        df = pd.concat([df, temp_df], ignore_index=True)
                        number_times+= 1
            #print(month[imonth],number_times, df)
            
            # This 'if' block needs to be indented one level back
            # to be *outside* the inner 'for i' loop
            # but *inside* the 'for imonth' loop
            if number_times > 0:
                write_header = not os.path.exists(output_file_name)
                if write_header:
                   print("File not found, writing new file with header.")
                   df.to_csv(output_file_name, index=False, mode='a', float_format='%g')
                else:
                   df.to_csv(output_file_name, index=False, mode='a', header = False, float_format='%g')
            
            # Reset number_times *after* writing, so it's fresh for the next month
            number_times = 0

#TODO: Convert csv file to zarr file.
#df = pd.read_csv('st_petersburg_surge_test.csv')
#df.to_zarr("zarr_surge_data_test.csv", mode='w', chunksize=(1000, None))
#Creating Zarr file

#
#Creating Zarr file
'''
zarr_file_name = 'zarr_input_data_st_petersburg_surge'
if os.path.exists(zarr_file_name):
    cmd = "rm -rf " + zarr_file_name
    os.system(cmd)
zarr_file_name = 'zarr_label_data_st_petersburg_surge'
if os.path.exists(zarr_file_name):
    cmd = "rm -rf " + zarr_file_name
    os.system(cmd)
df_training = zarr.create((number_times,6), chunks=(5,6), dtype=np.float32, store ='zarr_input_data_st_petersburg_surge')
df_label = zarr.create((number_times,6), chunks=(5,6), dtype=np.float32, store ='zarr_label_data_st_petersburg_surge')
string_index = df_winds.index.strftime('%Y-%m-%d %H %M %S')
for i in range(0,number_times):
    df_training[i, 0] = df.air_temp.values[i]
    df_training[i, 1] = df.air_pressure.values[i]
    df_training[i, 2] = df.wind_speed.values[i]
    df_training[i, 3] = df.wind_direction.values[i]
    df_training[i, 4] = df.wind_gust.values[i]
    df_training[i, 5] = df.water_level.values[i]
'''
#remove surge_data_test.csv
#cmd = "rm -f surge_data_test.csv"
#os.sytem(cmd)
