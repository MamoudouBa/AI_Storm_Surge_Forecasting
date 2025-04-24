#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
from py_noaa import coops
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
#
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

year = ['2002', '2003']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24',
         '25', '26','27', '28', '29', '30', '31']
number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

st_petersburg  = nc.Station(8726520)
ft_myers  = nc.Station(8725520)
old_port_tampa  = nc.Station(8726607)
clear_water_beatch  = nc.Station(8727520)

file_name = "/contrib/Mamoudou.Ba/st_petersburg_surge_test.csv"
if os.path.exists(file_name):
   cmd = "rm " + file_name
   os.system(cmd)

df_output = pd.DataFrame()
number_times = 0
for iyear in range(0,1):
     if leap_year(iyear):number_of_day_in_a_month[1] = 29
     for imonth in range (5,6):
             #beginDate = year[iyear] + month[imonth] + day[len(number_of_day_in_a_month(imonth) - 1)]
             beginDate = year[iyear] + month[imonth] + day[0]
             #endDate = year[iyear] + month[imonth] + day[len(number_of_day_in_a_month(imonth)]
             endDate = year[iyear] + month[imonth] + day[4]
             df_water_levels = st_petersburg.get_data(
               begin_date=beginDate,
               end_date=endDate,
               product="water_level",
               datum="MLLW",
               units="metric",
               time_zone="gmt")
             df_air_temp = st_petersburg.get_data(
               begin_date=beginDate,
               end_date=endDate,
               product="air_temperature",
               units="metric",
               time_zone="gmt")

             df_winds = st_petersburg.get_data(
               begin_date=beginDate,
               end_date=endDate,
               product="wind",
               units="metric",
               time_zone="gmt")
             df_air_pressure = st_petersburg.get_data(
               begin_date=beginDate,
               end_date=endDate,
               product="air_pressure",
               units="metric",
               time_zone="gmt")
#
     #print(len(df_winds.values), len(df_air_pressure))
     df_air_pressure = df_air_pressure.dropna()
     df_winds = df_winds.dropna()
     df_air_temp = df_air_temp.dropna()
     df_air_pressure = df_air_pressure.dropna()
     # Content of each variable array
     #air pressure: [pressure obs, 'string flag']
     #surface wind: [wind speed, wind direction in degrees, wind direction in words, gust speed, string flag]
     #air temperature: [temperature ons, string flag]
     #water level: [water level obs, sigma, qc flag, string flag]
     df = pd.DataFrame()
     for i in range(len(df_air_temp)):
        # Parse the time series index to string index
                 string_index = df_water_levels.index.strftime('%Y-%m-%d %H %M %S')
                 # Extract hour and minute
                 datetime_split = string_index[i].split(' ')
                 hour = datetime_split[1][0:2]
                 minute = datetime_split[2][0:2]
                 mn = int(minute)
                 if  mn == 0:
                     #If using all scalar values, you must pass an index
                     #reference: https://stackoverflow.com/questions/17839973
                     #/constructing-dataframe-from-values-in-variables-yields-valueerror-if-using-all
                     #
                     temp_df = pd.DataFrame({'timestamp':string_index[i],'air_temp':[df_air_temp.values[i][0]],
                       'air_pressure':[df_air_pressure.values[i][0]],'wind_speed':[df_winds.values[i][0]],
                       'wind_direction':[df_winds.values[i][1]],
                       'wind_gust':[df_winds.values[i][3]],'water_level': [df_water_levels.values[i][0]]})
                     df = pd.concat([df, temp_df], ignore_index=True)
                     number_times+= 1
                 df_output = df
     if number_times > 0:
        df_output.to_csv('/contrib/Mamoudou.Ba/st_petersburg_surge_test.csv', index=False, mode='a', float_format='%g')
        print("number_times is",number_times + 1)
df = pd.read_csv('/contrib/Mamoudou.Ba/st_petersburg_surge_test.csv')
#
#Creating Zarr file
zarr_file_name = '/contrib/Mamoudou.Ba/zarr_input_data_st_petersburg_surge'
if os.path.exists(zarr_file_name):
    cmd = "rm -rf " + zarr_file_name
    os.system(cmd)
zarr_file_name = '/contrib/Mamoudou.Ba/zarr_label_data_st_petersburg_surge'
if os.path.exists(zarr_file_name):
    cmd = "rm -rf " + zarr_file_name
    os.system(cmd)
df_training = zarr.create((number_times,6), chunks=(5,6), dtype=np.float32, store = \
                           '/contrib/Mamoudou.Ba/zarr_input_data_st_petersburg_surge')
df_label = zarr.create((number_times,6), chunks=(5,6), dtype=np.float32, store = \
                           '/contrib/Mamoudou.Ba/zarr_label_data_st_petersburg_surge')
string_index = df_winds.index.strftime('%Y-%m-%d %H %M %S')
for i in range(0,number_times):
    df_training[i, 0] = df.air_temp.values[i]
    df_training[i, 1] = df.air_pressure.values[i]
    df_training[i, 2] = df.wind_speed.values[i]
    df_training[i, 3] = df.wind_direction.values[i]
    df_training[i, 4] = df.wind_gust.values[i]
    df_training[i, 5] = df.water_level.values[i]

#removng the csv filee
#cmd = "rm -f /contrib/Mamoudou.Ba/st_petersburg_surge.csv"
#os.sytem(cmd)
