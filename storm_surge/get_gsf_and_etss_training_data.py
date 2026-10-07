#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

#https://www.geeksforgeeks.org/gated-recurrent-unit-networks/

#Import libraries
#
import numpy as np
import pandas as pd
import csv
from datetime import datetime
# Opening input data
import pandas as pd
import numpy as np
import pygrib
import warnings
#
# Remove the csv file if exists
# to avoid appending data to the old ones.
#
# etss_co_csv_20250421_cactus_20250812.tz
# Loop over datetime
# Author: Mamoudou Ba - June 2025-
import os

def leap_year(y):
    if y % 400 == 0:
        return True
    if y % 100 == 0:
        return False
    if y % 4 == 0:
        return True
    else:
        return False

# Initializing data arrays
#
year = ['2021','2022', '2023', '2024', '2025']
month = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
day = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25',
         '26', '27', '28', '29', '30', '31']

number_of_day_in_a_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
forecast_hour = ['00','01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12',
         '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23']
model_cycle = ['00', '06', '12', '18']
# Control the forecast times selected for each model cycle
# For example, for each cycle, the first 6 forecast hours are selected
# For example, for cycle 00: forecast times = 0, 1, 2, 3, 4, 5; 
#              for cycle 06: forecast times = 6, 7, 8, 9, 10, 11
start_time = 0
end_time = 0

# Station ID: 8726520 (St Petersburg
# Station coodinates
# lat_cell_size of etss subgrid = 13km/111.km = 0.117  (13 km)
# lontitude_cell_size of etss subgrid = 13km/111.km:cos(latitude) = 0.117  (13 km)
# 13km = 0.117 degree
# lat_min = 2S
# I_index = (latitude - lat_min)/0.117 + 1
# lon_min = -170
# J_index = ((longitude - lon_min)/lontitude_cell_size) + 1
# St Petersburg:
latitude = 27.78
longitude = -82.63
I_index = 221
J_index = 661
float_variables = [0.0,0.0,0.0,0.0,0.0,0.0]
current_timestamp = ''
data_rows = []
#Check if csv file exist, if so remove it
if os.path.exists('st_petersburg_training_data_gfs_etss_2022_2024.csv'):
   cmd = 'rm -f st_petersburg_training_data_gfs_etss_2022_2024.csv'
   os.system(cmd)
#Main loop start here, looping on number of years, months and days
for iyear in range(0,4):
  if leap_year(iyear):
       number_of_day_in_a_month[1] = 29
  else:
        number_of_day_in_a_month[1] = 28

  if year[iyear] == '2025':
       imonth_end = 6
#  if year[iyear] == '2022':
#       imonth_start = 8
#       iday_start = 1

  imonth_start = 0
  imonth_end = 12
  iday_start = 0
  print("year ", year[iyear])
  if year[iyear] == '2022':
     imonth_start = 8
     iday_start = 1
# Station ID: 8726520 (St Petersburg
# Station coodinates
# Open the CSV file in write mode
  with open('st_petersburg_training_data_gfs_etss_2022_2024.csv', 'a', newline='') as f:
   csv_writer = csv.writer(f)
  # Write a header row (optional)
   if iyear == 0: csv_writer.writerow(['timestamp','air_temp','air_pressure','wind_speed','wind_direction','wind_gust','water_level'])
   for imonth in range(imonth_start, imonth_end):
   #for imonth in range(0, 12):
  #Looping over days of the month
    #for iday in range(7, number_of_day_in_a_month[imonth]):
    #for icycle in range(0,4):
    #for iday in range(1,2):
    #  date_time_str = year[iyear] + month[imonth] + day[iday]
    #  print('date_time_str: ', date_time_str)
    fct_00 = -1
    fct_06 = 4
    fct_12 = 10
    fct_18 = 16
      #for j in range(0,6):
      #for j in range(0,6):
         #for icycle in range(0,4):
    for iday in range(0,number_of_day_in_a_month[imonth]):
           date_time_str = year[iyear] + month[imonth] + day[iday]
           #print('date_time_str: ', date_time_str)
# cycle 00z
           icycle = 0
           j = 0
           start_time = 0
           end_time = 6
           hour_str = 0
           for fct in range(start_time,end_time): 
             path_etss_file = '/scratch4/STI/mdl-sti/Mamoudou.Ba/storm_surge/etss_data/' + date_time_str +'/t' + model_cycle[icycle] + 'z_csv/8726520.csv'
             #print(path_etss_file)
             hour_str = str(fct)
             if fct < 10: hour_str = "0" + str(fct)
           #print(date_time_str,'path: ',path_etss_file)
             if os.path.exists(path_etss_file):
               with open(path_etss_file, 'r', newline='') as csvfile:
    # Create a CSV reader object, specifying the delimiter (comma by default)
                 csv_reader = csv.reader(csvfile)

    # Optionally, skip the header row if present
                 header = next(csv_reader) # Reads the first row (header)
    # Iterate over each row in the CSV file
            
                 for row in csv_reader:
        # Each 'row' is a list of strings representing the values in that row
               #print(f"Time: {row[0]}, TIDE: {row[1]}, OB: {row[2]}, SURGE: {row[3]}, BIAS: {row[2]}, TWL: {row[2]}")
                   if f"{row[5]}" == "9999.000": continue
                   #print('water level ', f"{row[5]}")
                   date_str = f"{row[0]}"
                   yr = date_str[0:4]
                   mm = date_str[4:6]
                   dd = date_str[6:8]
                   minutes = date_str[8:10]
                   if minutes != '00':continue
                   time_str = f"Time: {row[0]}"
                   etss_cycle = time_str[14:16]
                   second = time_str[16:18]
                   if second != "00": contine
                   #print('time, etss : ', time_str,etss_cycle)
                   date_str = yr + mm + dd
                   if int(yr) >= 2022:
                      pgrb2 = 'z.pgrb2f0'
                   else:
                      pgrb2 = 'z.pgrb2.0p25.f0'

                   gfs_gust_file_name = 'gfs_27km_gust_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + pgrb2 + forecast_hour[fct] +  '.grib2'
                   gfs_sflux_file_name = 'gfs_13km_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + 'z.sfluxgrbf0' + forecast_hour[fct] +  '.grib2'
                 #           + 'z.sfluxgrbf020.f' +  forecast_hour[fct]  + '.grib2'
                   #if not os.path.exists(gfs_gust_file_name): continue
                   #if not os.path.exists(gfs_sflux_file_name): continue
                   if os.path.exists(gfs_sflux_file_name) and os.path.exists(gfs_gust_file_name):
                      #print(gfs_gust_file_name, gfs_sflux_file_name)
                      gust_field = pygrib.open(gfs_gust_file_name)
                      met_obs_fields = pygrib.open(gfs_sflux_file_name)
                      #print(icycle, gfs_sflux_file_name, os.path.getsize(gfs_sflux_file_name))
                      if os.path.getsize(gfs_sflux_file_name) < 5000000: continue
                 # Extract gfs variables
                      temp = met_obs_fields.message(2).values[:,:] # Kelvin
                      temp = temp - 273.15                         #Celsius
                      presure = met_obs_fields.message(1).values[:,:] # pascal 
           # Compute wind speed vector at 10 m above the surface
                      U_comp = met_obs_fields.message(3).values[:,:] # m/s
                      V_comp = met_obs_fields.message(4).values[:,:]  # m/s
                      u = U_comp[I_index, J_index]
                      v = V_comp[I_index, J_index]
                 #Compute wind speed and direction at surface
                      value = u*u + v*v
                      V_surface = np.sqrt(value)
                      V_surface = round(V_surface,2)
                      WDIR = np.arctan2(v,u) * (180/np.pi) + 180. # degree
                      WDIR = round(WDIR,2)

           # Extract gust values
                      gust_grid = gust_field.message(1).values[:,:]
                      gust = round(gust_grid[I_index, J_index],2) 
                      fct_str = str(fct)
                      if fct < 10: fct_str = '0' + str(fct)
                      current_timestamp = time_str[6:14] + hour_str + '00'  
                      temp = round(temp[I_index, J_index],2)
                      if(temp > 330.0): continue
                      pres = round(presure[I_index, J_index]/100.0,2)
                      water_level = f"{row[5]}"
                      wl = float(water_level)
                      wl = round(wl, 2)
                      float_variables = [temp,pres, V_surface,WDIR, gust, wl]
                      #print('float_variables  ', float_variables)
                      #print(current_timestamp, float_variables)
                      #data_rows = np.array[f"Time: {row[0]}",temp[I_index, J_index], presure[I_index, J_index]/100.0, V_surface,WDIR, gust, f"{row[5]}"]
                      #data_rows.append([current_timestamp]+ float_variables)
                      #print(icycle, j,fct)
                      #print('Writing cycle 00z data', [current_timestamp]+ float_variables)
                   datetime_str = ""
                   for dt in [current_timestamp]:
                       datetime_str = dt

# Output: 2025-10-27 08:30:00, 2025-10-27 09:15:45, 2025-10-27 10:05:10
                  # 1. Parse the string into a datetime object
                   input_string = datetime_str
                   dt_object = datetime.strptime(input_string, "%Y%m%d%H%M")

                  # 2. Convert the datetime object to a float (Unix timestamp)
                   timestamp_float = dt_object.timestamp()

                   #print(f"The float value of '{input_string}' is: {timestamp_float}")
                   # Your float timestamp (e.g., the current time)
                   #float_timestamp = 1760000270.5  # Represents Oct 10, 2025 at 10:57:50 AM EDT

                   # 1. Convert the float to a datetime object
                   dt_object = datetime.fromtimestamp(timestamp_float)
                   formatted_string = dt_object.strftime('%Y-%m-%d %H %M %S')
                   #print(f"Formatted String: {formatted_string}")
                   print([formatted_string] + float_variables)
                   csv_writer.writerow([formatted_string] + float_variables)

           
             else:
               continue
           j +=1
#cycle 06z
           j = 0
           start_time = 5
           end_time = 12
           hour_str = 0
           icycle = 1
           for fct in range(start_time,end_time):
             path_etss_file = '/scratch4/STI/mdl-sti/Mamoudou.Ba/storm_surge/etss_data/' + date_time_str +'/t' + model_cycle[icycle] + 'z_csv/8726520.csv'
             #print(path_etss_file)
             hour_str = str(fct)
             if fct < 10: hour_str = "0" + str(fct)
           #print(date_time_str,'path: ',path_etss_file)
             if os.path.exists(path_etss_file) and os.path.getsize(path_etss_file) == 0: continue
             if os.path.exists(path_etss_file):
               with open(path_etss_file, 'r', newline='') as csvfile:
    # Create a CSV reader object, specifying the delimiter (comma by default)
                 csv_reader = csv.reader(csvfile)

    # Optionally, skip the header row if present
                 header = next(csv_reader) # Reads the first row (header)
    # Iterate over each row in the CSV file

                 for row in csv_reader:
        # Each 'row' is a list of strings representing the values in that row
               #print(f"Time: {row[0]}, TIDE: {row[1]}, OB: {row[2]}, SURGE: {row[3]}, BIAS: {row[2]}, TWL: {row[2]}")
                   if f"{row[5]}" == "9999.000": continue
                   date_str = f"{row[0]}"
                   yr = date_str[0:4]
                   mm = date_str[4:6]
                   dd = date_str[6:8]
                   minutes = date_str[8:10]
                   if minutes != '00':continue
                   time_str = f"Time: {row[0]}"
                   etss_cycle = time_str[14:16]
                   second = time_str[16:18]
                   if second != "00": contine
                   #print('time, etss : ', time_str,etss_cycle)
                   date_str = yr + mm + dd
                   gfs_gust_file_name = 'gfs_27km_gust_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + pgrb2 + forecast_hour[fct] +  '.grib2'
                   gfs_sflux_file_name = 'gfs_13km_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + 'z.sfluxgrbf0' + forecast_hour[fct] +  '.grib2'
                 #           + 'z.sfluxgrbf020.f' +  forecast_hour[fct]  + '.grib2'
                   if not os.path.exists(gfs_gust_file_name): continue
                   if not os.path.exists(gfs_sflux_file_name): continue
                   gust_field = pygrib.open(gfs_gust_file_name)
                   met_obs_fields = pygrib.open(gfs_sflux_file_name)
                   if os.path.getsize(gfs_sflux_file_name) < 5000000: continue
                 # Extract gfs variables
                   temp = met_obs_fields.message(2).values[:,:] # Kelvin
                   temp = temp - - 273.15 # Celsius
                   presure = met_obs_fields.message(1).values[:,:] # pascal
           # Compute wind speed vector at 10 m above the surface
                   U_comp = met_obs_fields.message(3).values[:,:] # m/s
                   V_comp = met_obs_fields.message(4).values[:,:]  # m/s
                   u = U_comp[I_index, J_index]
                   v = V_comp[I_index, J_index]
                 #Compute wind speed and direction at surface
                   value = u*u + v*v
                   V_surface = np.sqrt(value)
                   V_surface = round(V_surface,2)
                   WDIR = np.arctan2(v,u) * (180/np.pi) + 180. # degree
                   WDIR = round(WDIR,2)

           # Extract gust values
                   gust_grid = gust_field.message(1).values[:,:]
                   gust = round(gust_grid[I_index, J_index],2)
                   fct_str = str(fct)
                   if fct < 10: fct_str = '0' + str(fct)
                   current_timestamp = time_str[6:14] + hour_str + '00'
                   temp = round(temp[I_index, J_index],2)
                   if(temp > 330.0): continue
                   #print('***** temp **** :', tmp)
                   pres = round(presure[I_index, J_index]/100.0,2)
                   water_level = f"{row[5]}"
                   wl = float(water_level)
                   wl = round(wl, 2)
                   float_variables = [temp,pres, V_surface,WDIR, gust, wl]
                   # Convert each timestamp to a string and join them
                   datetime_str = ""
                   for dt in [current_timestamp]:
                       datetime_str = dt
                        
# Output: 2025-10-27 08:30:00, 2025-10-27 09:15:45, 2025-10-27 10:05:10
                  # 1. Parse the string into a datetime object
                   input_string = datetime_str
                   dt_object = datetime.strptime(input_string, "%Y%m%d%H%M")

                  # 2. Convert the datetime object to a float (Unix timestamp)
                   timestamp_float = dt_object.timestamp()

                   #print(f"The float value of '{input_string}' is: {timestamp_float}")
                   # Your float timestamp (e.g., the current time)
                   #float_timestamp = 1760000270.5  # Represents Oct 10, 2025 at 10:57:50 AM EDT

                   # 1. Convert the float to a datetime object
                   dt_object = datetime.fromtimestamp(timestamp_float)
                   formatted_string = dt_object.strftime('%Y-%m-%d %H %M %S')
                   #print(f"Formatted String: {formatted_string}")
                   print([formatted_string] + float_variables)
                   csv_writer.writerow([formatted_string] + float_variables)

             else:
               continue
           j +=1


#  cycle 12z
           icycle = 2
           j = 0
           start_time = 11
           end_time = 17
           hour_str = 0
           for fct in range(start_time,end_time):
             path_etss_file = '/scratch4/STI/mdl-sti/Mamoudou.Ba/storm_surge/etss_data/' + date_time_str +'/t' + model_cycle[icycle] + 'z_csv/8726520.csv'
             #print(path_etss_file)
             hour_str = str(fct)
             if fct < 10: hour_str = "0" + str(fct)
           #print(date_time_str,'path: ',path_etss_file)
             if os.path.exists(path_etss_file):
               with open(path_etss_file, 'r', newline='') as csvfile:
    # Create a CSV reader object, specifying the delimiter (comma by default)
                 csv_reader = csv.reader(csvfile)

    # Optionally, skip the header row if present
                 header = next(csv_reader) # Reads the first row (header)
    # Iterate over each row in the CSV file

                 for row in csv_reader:
        # Each 'row' is a list of strings representing the values in that row
               #print(f"Time: {row[0]}, TIDE: {row[1]}, OB: {row[2]}, SURGE: {row[3]}, BIAS: {row[2]}, TWL: {row[2]}")
                   if f"{row[5]}" == "9999.000": continue
                   date_str = f"{row[0]}"
                   yr = date_str[0:4]
                   mm = date_str[4:6]
                   dd = date_str[6:8]
                   minutes = date_str[8:10]
                   if minutes != '00':continue
                   time_str = f"Time: {row[0]}"
                   etss_cycle = time_str[14:16]
                   second = time_str[16:18]
                   if second != "00": contine
                   #print('time, etss : ', time_str,etss_cycle)
                   date_str = yr + mm + dd
                   gfs_gust_file_name = 'gfs_27km_gust_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + pgrb2 + forecast_hour[fct] +  '.grib2'
                   gfs_sflux_file_name = 'gfs_13km_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + 'z.sfluxgrbf0' + forecast_hour[fct] +  '.grib2'
                 #           + 'z.sfluxgrbf020.f' +  forecast_hour[fct]  + '.grib2'
                   if not os.path.exists(gfs_gust_file_name): continue
                   if not os.path.exists(gfs_sflux_file_name): continue
                   gust_field = pygrib.open(gfs_gust_file_name)
                   met_obs_fields = pygrib.open(gfs_sflux_file_name)
                   if os.path.getsize(gfs_sflux_file_name) < 5000000: continue
                 # Extract gfs variables
                   temp = met_obs_fields.message(2).values[:,:] # Kelvin
                   temp = temp - 273.15                         # Celsius
                   presure = met_obs_fields.message(1).values[:,:] # pascal
           # Compute wind speed vector at 10 m above the surface
                   U_comp = met_obs_fields.message(3).values[:,:] # m/s
                   V_comp = met_obs_fields.message(4).values[:,:]  # m/s
                   u = U_comp[I_index, J_index]
                   v = V_comp[I_index, J_index]
                 #Compute wind speed and direction at surface
                   value = u*u + v*v
                   V_surface = np.sqrt(value)
                   V_surface = round(V_surface,2)
                   WDIR = np.arctan2(v,u) * (180/np.pi) + 180. # degree
                   WDIR = round(WDIR,2)

           # Extract gust values
                   gust_grid = gust_field.message(1).values[:,:]
                   gust = round(gust_grid[I_index, J_index],2)
                   fct_str = str(fct)
                   if fct < 10: fct_str = '0' + str(fct)
                   current_timestamp = time_str[6:14] + hour_str + '00'
                   temp = round(temp[I_index, J_index],2)
                   if(temp > 330.0): continue
                   pres = round(presure[I_index, J_index]/100.0,2)
                   water_level = f"{row[5]}"
                   wl = float(water_level)
                   wl = round(wl, 2)
                   float_variables = [temp,pres, V_surface,WDIR, gust, wl]
                   #print(current_timestamp, float_variables)
                   #data_rows = np.array[f"Time: {row[0]}",temp[I_index, J_index], presure[I_index, J_index]/100.0, V_surface,WDIR, gust, f"{row[5]}"]
                   #data_rows.append([current_timestamp]+ float_variables)
                   #print('Writing cycle 12z data', [current_timestamp]+ float_variables)
                   datetime_str = ""
                   for dt in [current_timestamp]:
                       datetime_str = dt

# Output: 2025-10-27 08:30:00, 2025-10-27 09:15:45, 2025-10-27 10:05:10
                  # 1. Parse the string into a datetime object
                   input_string = datetime_str
                   dt_object = datetime.strptime(input_string, "%Y%m%d%H%M")

                  # 2. Convert the datetime object to a float (Unix timestamp)
                   timestamp_float = dt_object.timestamp()

                   #print(f"The float value of '{input_string}' is: {timestamp_float}")
                   # Your float timestamp (e.g., the current time)
                   #float_timestamp = 1760000270.5  # Represents Oct 10, 2025 at 10:57:50 AM EDT

                   # 1. Convert the float to a datetime object
                   dt_object = datetime.fromtimestamp(timestamp_float)
                   formatted_string = dt_object.strftime('%Y-%m-%d %H %M %S')
                   #print(f"Formatted String: {formatted_string}")
                   print([formatted_string] + float_variables)
                   csv_writer.writerow([formatted_string] + float_variables)

             else:
               continue
           j +=1
#cycle 18z
           icycle = 3
           j = 0
           start_time = 17
           end_time = 24
           hour_str = 0
           for fct in range(start_time,end_time):
             path_etss_file = '/scratch4/STI/mdl-sti/Mamoudou.Ba/storm_surge/etss_data/' + date_time_str +'/t' + model_cycle[icycle] + 'z_csv/8726520.csv'
             #print(path_etss_file)
             hour_str = str(fct)
             if fct < 10: hour_str = "0" + str(fct)
           #print(date_time_str,'path: ',path_etss_file)
             if os.path.exists(path_etss_file):
               with open(path_etss_file, 'r', newline='') as csvfile:
    # Create a CSV reader object, specifying the delimiter (comma by default)
                 csv_reader = csv.reader(csvfile)

    # Optionally, skip the header row if present
                 header = next(csv_reader) # Reads the first row (header)
    # Iterate over each row in the CSV file

                 for row in csv_reader:
        # Each 'row' is a list of strings representing the values in that row
               #print(f"Time: {row[0]}, TIDE: {row[1]}, OB: {row[2]}, SURGE: {row[3]}, BIAS: {row[2]}, TWL: {row[2]}")
                   if f"{row[5]}" == "9999.000": continue
                   date_str = f"{row[0]}"
                   yr = date_str[0:4]
                   mm = date_str[4:6]
                   dd = date_str[6:8]
                   minutes = date_str[8:10]
                   if minutes != '00':continue
                   time_str = f"Time: {row[0]}"
                   etss_cycle = time_str[14:16]
                   second = time_str[16:18]
                   if second != "00": contine
                   #print('time, etss : ', time_str,etss_cycle)
                   date_str = yr + mm + dd
                   gfs_gust_file_name = 'gfs_27km_gust_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + pgrb2 + forecast_hour[fct] +  '.grib2'
                   gfs_sflux_file_name = 'gfs_13km_data/' + date_str + '/gfs.t' + \
                          model_cycle[icycle] + 'z.sfluxgrbf0' + forecast_hour[fct] +  '.grib2'
                 #           + 'z.sfluxgrbf020.f' +  forecast_hour[fct]  + '.grib2'
                   if not os.path.exists(gfs_gust_file_name): continue
                   if not os.path.exists(gfs_sflux_file_name): continue
                   gust_field = pygrib.open(gfs_gust_file_name)
                   met_obs_fields = pygrib.open(gfs_sflux_file_name)
                   if os.path.getsize(gfs_sflux_file_name) < 5000000: continue
                   #print('Files are opened')
                 # Extract gfs variables
                   temp = met_obs_fields.message(2).values[:,:] # Kelvin
                   presure = met_obs_fields.message(1).values[:,:] # pascal
           # Compute wind speed vector at 10 m above the surface
                   U_comp = met_obs_fields.message(3).values[:,:] # m/s
                   V_comp = met_obs_fields.message(4).values[:,:]  # m/s
                   u = U_comp[I_index, J_index]
                   v = V_comp[I_index, J_index]
                 #Compute wind speed and direction at surface
                   value = u*u + v*v
                   V_surface = np.sqrt(value)
                   V_surface = round(V_surface,2)
                   WDIR = np.arctan2(v,u) * (180/np.pi) + 180. # degree
                   WDIR = round(WDIR,2)
                   #print('WDIR   ', WDIR, 'temp ', temp, 'presure ', presure)

           # Extract gust values
                   gust_grid = gust_field.message(1).values[:,:]
                   gust = round(gust_grid[I_index, J_index],2)
                   fct_str = str(fct)
                   if fct < 10: fct_str = '0' + str(fct)
                   current_timestamp = time_str[6:14] + hour_str + '00'
                   temp = temp - 273.15
                   temp = round(temp[I_index, J_index],2)
                   if(temp > 330.0): continue
                   pres = round(presure[I_index, J_index]/100.0,2)
                   water_level = f"{row[5]}"
                   wl = float(water_level)
                   wl = round(wl, 2)
                   float_variables = [temp,pres, V_surface,WDIR, gust, wl]
                   #print(current_timestamp, float_variables)
                   #data_rows = np.array[f"Time: {row[0]}",temp[I_index, J_index], presure[I_index, J_index]/100.0, V_surface,WDIR, gust, f"{row[5]}"]
                   #data_rows.append([current_timestamp]+ float_variables)
                   #print('Writing cycle 18z data', [current_timestamp]+ float_variables)
                   datetime_str = ""
                   for dt in [current_timestamp]:
                       datetime_str = dt

# Output: 2025-10-27 08:30:00, 2025-10-27 09:15:45, 2025-10-27 10:05:10
                  # 1. Parse the string into a datetime object
                   input_string = datetime_str
                   dt_object = datetime.strptime(input_string, "%Y%m%d%H%M")

                  # 2. Convert the datetime object to a float (Unix timestamp)
                   timestamp_float = dt_object.timestamp()

                   #print(f"The float value of '{input_string}' is: {timestamp_float}")
                   # Your float timestamp (e.g., the current time)
                   #float_timestamp = 1760000270.5  # Represents Oct 10, 2025 at 10:57:50 AM EDT

                   # 1. Convert the float to a datetime object
                   dt_object = datetime.fromtimestamp(timestamp_float)
                   formatted_string = dt_object.strftime('%Y-%m-%d %H %M %S')
                   #print(f"Formatted String: {formatted_string}")
                   print([formatted_string] + float_variables)
                   csv_writer.writerow([formatted_string] + float_variables)


             else:
               continue
           j +=1

      #         print(model_cycle[icycle],fct,date_str, float_variables)
           #else:
           #    continue
           #print(model_cycle[icycle],fct,time_str,current_timestamp)
# pandas format
#timestamp,air_temp,air_pressure,wind_speed,wind_direction,wind_gust,water_level
                 #print(f"Time: {row[0]}",temp[I_index, J_index], presure[I_index, J_index]/100.0, V_surface,WDIR, gust, f"{row[5]}")

                 #print ('******* ',gfs_gust_file_name)
                 #print ('+++++++: ',gfs_sflux_file_name)
# Open the CSV file in write mode

