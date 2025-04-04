#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
from py_noaa import coops
#https://api.tidesandcurrents.noaa.gov/api/prod/
#
from datetime import datetime
import noaa_coops as nc
import pandas as pd
from noaa_coops import Station
import noaa_coops as nc
#import xarray as xr
import zarr
import os

st_petersburg  = nc.Station(8726520)
ft_myers  = nc.Station(8725520)
old_port_tampa  = nc.Station(8726607 )
clear_water_beatch  = nc.Station(8727520)

print("start retrieving data")
df_water_levels = st_petersburg.get_data(
       begin_date="20020401",
       end_date="20020403",
       product="water_level",
       datum="MLLW",
       units="metric",
       time_zone="gmt")
df_air_temp = st_petersburg.get_data(
       begin_date="20020401",
       end_date="20020403",
       product="air_temperature",
       units="metric",
       time_zone="gmt")

df_winds = st_petersburg.get_data(
       begin_date="20020401",
       end_date="20020403",
       product="wind",
       units="metric",
       time_zone="gmt")
df_air_pressure = st_petersburg.get_data(
       begin_date="20020401",
       end_date="20020403",
       product="air_pressure",
       units="metric",
       time_zone="gmt")

#
print(df_air_pressure)
if os.path.exists("st_petersburg_surge_test.csv"):
    cmd = "rm -rf st_petersburg_surge_test.csv"
    os.system(cmd)

#print('printing the first index ',df_air_temp.index[0])
#for i in range(len(df_air_temp)):
df = pd.DataFrame()
for i in range(len(df_air_temp)):
        # Convert index to datetim
        # Parse the string to a datetime object
        # index to datetime
        date_time = pd.to_datetime(df_air_temp.index[i], format='%Y-%m-%d %H %M %S')

        # Extract hour and minute
        hour = date_time.hour
        minute = date_time.minute
        if minute != 0: continue
        #print(df_air_temp.index[i], df_air_temp.v[i],df_water_levels.v[i])
        temp_df = pd.DataFrame({'timestamp':df_air_temp.index[i],'air_temp':df_air_temp.v,
            'air_pressure':df_air_pressure.v,'wind_speed':df_winds.s,
            'wind_direction':df_winds.d,
            'wind_gust':df_winds.g,'water_level': df_water_levels.v})
        temp_df = temp_df.dropna()
        df = pd.concat([df, temp_df], ignore_index=True)
# Write to CSV file
#df = pd.DataFrame(data=df)
#empty_cols = (df.isnull().all()) | (df.eq('').all())
#df_cleaned = df.dropna(axis=1, how='all')
#df_cleaned = df.loc[:, ~empty_cols]
#        temp_df = pd.DataFrame({'timestamp':df_air_temp.index[i],'air_temp':df_air_temp.v,
#            'air_pressure':df_air_pressure.v,'wind_speed':df_winds.s,
#            'wind_direction':df_winds.d,
#            'wind_gust':df_winds.g,'water_level': df_water_levels.v})
#df_filtered = df.dropna()
#print(df_filtered)
df.to_csv('st_petersburg_surge_test.csv', index=False, mode='a', float_format='%g')
#df_filtered.to_csv('st_petersburg_surge_test.csv', index=False, mode='a', float_format='%g')
#df_filtered.to_csv('surge_data_test.csv', index=False)

# Convert cvs file to zarr storage


#df = pd.read_csv('surge_data_test.csv')
#df.to_zarr("zarr_surge_data_test.csv", mode='w', chunksize=(1000, None))

#remove surge_data_test.csv
#cmd = "rm -f surge_data_test.csv"
#os.sytem(cmd)
