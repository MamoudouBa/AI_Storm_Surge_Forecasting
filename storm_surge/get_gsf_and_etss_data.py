#!/scratch4/STI/mdl-sti/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python

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
model_cycle = ['00', '06', '12', '18']
# Control the forecast times selected for each model cycle
# For example, for each cycle, the first 6 forecast hours are selected
# For example, for cycle 00: forecast times = 0, 1, 2, 3, 4, 5; 
#              for cycle 06: forecast times = 6, 7, 8, 9, 10, 11
start_time = 0
end_time = 0
# Station ID: 8726520 (St Petersburg
# Station coodinates
latitude = 27.78
longitude = -82.63
I_index = 436
J_index = 896
float_variables = [0.0,0.0,0.0,0.0,0.0,0.0]
current_timestamp = ''
data_rows = []
#Check if csv file exist, if so remove it
if os.path.exists('st_petersburg_training_data.csv'):
   cmd = 'rm -f st_petersburg_training_data.csv'
   os.system(cmd)
#Main loop start here, looping on number of years, months and days
for iyear in range(0,3):
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
  if year[iyear] == '2025':
     imonth_end = 6
  if year[iyear] == '2022':
     imonth_start = 8
     iday_start = 1
# Station ID: 8726520 (St Petersburg
# Station coodinates
# Open the CSV file in write mode
# Open the CSV file in write mode
  with open('st_petersburg_training_data.csv', 'a', newline='') as f:
   csv_writer = csv.writer(f)
  # Write a header row (optional)
   if iyear == 0: csv_writer.writerow(['timestamp','air_temp','air_pressure','wind_speed','wind_direction','wind_gust','water_level'])
   #for imonth in range(imonth_start, imonth_end):
   for imonth in range(0, 12):
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

