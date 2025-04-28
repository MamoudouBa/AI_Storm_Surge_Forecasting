#!/contrib/Mamoudou.Ba/miniconda3/envs/mlaws2t/bin/python
import requests
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt, dates
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

import warnings
#
# Remove the csv file if exists
# to avoid appending data to the old ones.
if os.path.exists('/contrib/Mamoudou.Ba/surge_data_20220928.csv'):
    cmd = "rm -rf /contrib/Mamoudou.Ba/surge_data_20220928.csv" 
    os.system(cmd)

warnings.filterwarnings('ignore')
    ## Define your parameters 
number_times = 0
begin_date = '20220927'
end_date = '20220928'
station = '8725520'
datum = 'MHHW'
time_zone = 'gmt'
units = 'english'

    ## for plot title
station_name = 'Fort Myers'
storm_name = 'Hurricane Ian 2022'
    ## paste the URL from the URL builder
    ## this is the url for water level data
api = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date='+begin_date+'&end_date='+end_date+'&station='+station+'&product=water_level&datum='+datum+'&time_zone='+time_zone+'&units='+units+'&format=json#api'
api
    ## this is where the API is being accessed and data is being retrieved 
response = requests.get(api)
content = response.json()

content1 = content['metadata']
#content1

content2 = content['data']
#content2 
    ## create dataframe for data (content2)
df = pd.DataFrame(content2, columns = ['t', 'v'])

    ## rename columns
df.rename(columns = {'t':'datetime', 'v':'Water Level'}, inplace = True)

    ## set index to datetime column
df.set_index('datetime')
df.index = pd.to_datetime(df.datetime)

    ## drop other column called datetime
df.drop(labels = None, axis = 1, index = None, columns = ['datetime'], inplace = True)

    ## change type of data from object to float 
df['Water Level'] = df['Water Level'].astype(float)

df = df.dropna()


    ## create dataframe for data (content2)
df = pd.DataFrame(content2, columns = ['t', 'v'])

    ## rename columns
df.rename(columns = {'t':'datetime', 'v':'Water Level'}, inplace = True)

    ## set index to datetime column
df.set_index('datetime')
df.index = pd.to_datetime(df.datetime)

    ## drop other column called datetime
df.drop(labels = None, axis = 1, index = None, columns = ['datetime'], inplace = True)

    ## change type of data from object to float 
df['Water Level'] = df['Water Level'].astype(float)

df


    ## this is the url for air pressure data
# api2 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20220928&end_date=20220929&station=8725520&product=air_temperature&time_zone=gmt&units=english&format=json#api2'
#api2
api2 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date='+begin_date+'&end_date='+end_date+'&station='+station+'&product=air_temperature&datum='+datum+'&time_zone='+time_zone+'&units='+units+'&format=json#api2'


    ## this is where the API is being accessed and data is being retrieved
response = requests.get(api2)
contents = response.json()
#contents 

contents1 = contents['metadata']
#contents1

contents2 = contents['data']
#contents2

    ## create dataframe
df1 = pd.DataFrame(contents2, columns = ['t', 'v'])

    ## create dataframe
df1 = pd.DataFrame(contents2, columns = ['t', 'v'])

    ## rename columns 
df1.rename(columns = {'t':'datetime', 'v':'Air Temperature'}, inplace = True)

    ## set index to datetime column 
df1.set_index('datetime')
df1.index = pd.to_datetime(df1.datetime)

    ## drop other datetime column 
df1.drop(labels = None, axis = 1, index = None, columns = ['datetime'], inplace = True)

    ## some data is not available so this line of code is needed to account for the missing values
df1 = df1.replace('', np.nan, regex=True)

    ## change type from object to float
df1['Air Temperature'] = df1['Air Temperature'].astype(float)

df1 = df1.dropna()   


    ## this is the url for air pressure data
# api2 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20220928&end_date=20220929&station=8725520&product=air_pressure&time_zone=gmt&units=english&format=json#api2'
#api2
api2 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date='+begin_date+'&end_date='+end_date+'&station='+station+'&product=air_pressure&datum='+datum+'&time_zone='+time_zone+'&units='+units+'&format=json#api2'


    ## this is where the API is being accessed and data is being retrieved
response = requests.get(api2)
contents = response.json()
#contents 

contents1 = contents['metadata']
#contents1

contents2 = contents['data']
#contents2

    ## create dataframe
df2 = pd.DataFrame(contents2, columns = ['t', 'v'])

    ## create dataframe
df2 = pd.DataFrame(contents2, columns = ['t', 'v'])

    ## rename columns 
df2.rename(columns = {'t':'datetime', 'v':'Air Pressure'}, inplace = True)

    ## set index to datetime column 
df2.set_index('datetime')
df2.index = pd.to_datetime(df2.datetime)

    ## drop other datetime column 
df2.drop(labels = None, axis = 1, index = None, columns = ['datetime'], inplace = True)

    ## some data is not available so this line of code is needed to account for the missing values
df2 = df2.replace('', np.nan, regex=True)

    ## change type from object to float
df2['Air Pressure'] = df2['Air Pressure'].astype(float)

df2 = df2.dropna()   
    ## this is the url for wind data
# api3 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date=20220928&end_date=20220929&station=8725520&product=wind&time_zone=gmt&units=english&format=json#api3'
#api3
api3 = 'https://api.tidesandcurrents.noaa.gov/api/prod/datagetter?begin_date='+begin_date+'&end_date='+end_date+'&station='+station+'&product=wind&datum='+datum+'&time_zone='+time_zone+'&units='+units+'&format=json#api3'


    ## this is where the API is being accessed and data is being retrieved 
response = requests.get(api3)
contents = response.json()
#contents

contents1 = contents['metadata']
#contents1

contents2 = contents['data']
#contents2

    ## create dataframe
df3 = pd.DataFrame(contents2, columns = ['t', 's', 'd', 'dr', 'g'])

    ## rename columns
df3.rename(columns = {'t':'datetime', 's': 'speed', 'd': 'direction', 'dr':'cardinal direction', 'g':'gust'}, inplace = True)
    
    ## set index to datetime column
df3.set_index('datetime')
df3.index = pd.to_datetime(df3.datetime)
    
    ## drop other datetime column 
df3.drop(labels = None, axis = 1, index = None, columns = ['datetime'], inplace = True)
    
    ## some data is not available so this line of code is needed to account for the missing values
df3 = df3.replace('', np.nan, regex=True)
    
    ## change data to numeric data 
df3['direction'] = pd.to_numeric(df3['direction'])
df3['speed'] = pd.to_numeric(df3['speed'])

df_water_levels = df.dropna()
df_air_temp = df1.dropna()
df_air_temp = (df_air_temp - 32)*(5/9)
df_air_temp = df_air_temp.astype(float)
df_air_pressure = df2.dropna()
df_winds = df3.dropna()
print(df_winds)
# write to output file csv



df = pd.DataFrame()
for i in range(0,len(df_water_levels)):
        # Parse the time series index to string index
                 string_index = df_water_levels.index.strftime('%Y-%m-%d %H %M %S')
                 string_wind_index = df_winds.index.strftime('%Y-%m-%d %H %M %S')
                 string_air_temp_index = df_air_temp.index.strftime('%Y-%m-%d %H %M %S')
                 string_air_pressure_index = df_air_pressure.index.strftime('%Y-%m-%d %H %M %S')
                 if i<= len(string_wind_index) - 1 and string_wind_index[i] == string_index[i]: w_index = i
                 if i<= len(string_air_temp_index) - 1 and string_air_temp_index[i] == string_index[i]: t_index = i
                 if i<= len(string_air_pressure_index) - 1 and string_air_pressure_index[i] == string_index[i]: p_index = i
                 #print(string_air_temp_index[i])
                 # Check if date of meteorological observation matches that of water level observation
                 #if string_index[i] != w_index and string_index[i] != t_index and string_index[i] != p_index: continue
                 # Extract hour and minute
                 #print(i, len(df_air_temp), ' len of string_index ',len(string_index))
                 datetime_split = string_index[i].split(' ')

                 hour = datetime_split[1][0:2]
                 minute = datetime_split[2][0:2]
                 mn = int(minute)
                 #print(string_index[i],[df_air_temp.values[t_index][0]])
                 #print(w_index[i], t_index[i], p_index[i], string_index[i])
                 #print(hour,len(df_air_temp),len(df_winds), len(df_air_pressure), len(df_water_levels))
                 if  mn == 0:
                     #If using all scalar values, you must pass an index
                     #reference: https://stackoverflow.com/questions/17839973
                     #/constructing-dataframe-from-values-in-variables-yields-valueerror-if-using-all
                     #
                     # df = pd.DataFrame({'timestamp':[date],['air_temp':[temp obs], 'air_pressure':[bressure obs],
                     #                    'wind_speed':[wind speed], 'wind_direction':[wind direction obs],
                     #                    'wind_gust':[gust speed], 'water_level':[water level obs]]}
                     #
                      #print(i,string_index[i], w_index, t_index, p_index,df_air_temp.values[t_index][0])
                       #print(i,df_air_temp.values[t_index][0])
                       temp_df = pd.DataFrame({'timestamp':string_index[i],'air_temp':[df_air_temp.values[t_index][0]],
                          'air_pressure':[df_air_pressure.values[p_index][0]],'wind_speed':[df_winds.values[w_index][0]],
                          'wind_direction':[df_winds.values[w_index][1]],
                          'wind_gust':[df_winds.values[w_index][3]],'water_level': [df_water_levels.values[i][0]]})
                       df = pd.concat([df, temp_df], ignore_index=True)
                       #print(df)
                       number_times+= 1
                 #print(month[imonth],number_times)
if number_times > 0:
                 df.to_csv("surge_data_20220928.csv", index=False, mode='a', float_format='%g')


'''
    ## change the direction values from degrees to radians 
rad = np.deg2rad(df3['direction'])
    
    ## U and V define directions of arrows
U = -(df3['speed']) * np.sin(rad)
V = -(df3['speed']) * np.cos(rad)

plt.quiver(df3.index, df3['speed'], U, V)
plt.ylim(ymax = 70, ymin = 0)

plt.tight_layout()

    ## set date format for all following plots

date_form = dates.DateFormatter("%H:%M \n %m/%d")
    ## set default plot style
plt.style.use('ggplot')
    ## create a figure with subplots (rows, columns, figsize)
fig, axs = plt.subplots(3, 1, figsize = (9, 4.5 * 3))
plt.suptitle(station_name+' during '+storm_name, size = 20)

    ## assign data to axes and set titles, labels, and limits
    ## Water Level
axs[0].plot(df['Water Level'], color = 'royalblue')
axs[0].set_title('Water Level')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Water Level (ft above '+datum+')')
axs[0].set_ylim(ymax=max(df['Water Level']+1), ymin=min(df['Water Level'])-1)
axs[0].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[0].xaxis.set_major_formatter(date_form)
    ## Air Pressure 
axs[1].plot(df2['Air Pressure'], color = 'red')
axs[1].set_title('Air Pressure')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Air Pressure (mb)')
axs[1].set_ylim(ymax=max(df2['Air Pressure']+5), ymin=min(df2['Air Pressure'])-5)
axs[1].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[1].xaxis.set_major_formatter(date_form)

    ## Wind Speed and Wind Direction 
axs[2].plot(df3['speed'], color = 'green')
axs[2].quiver(df3.index, df3['speed'], U, V, color = 'black', width = 0.0007, headwidth = 6.5)
axs[2].set_title('Wind Speed and Wind Direction')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Wind Speed (knots)')
axs[2].set_ylim(ymax=max(df3['speed']+8), ymin=min(df3['speed'])-5)
axs[2].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[2].xaxis.set_major_formatter(date_form)

    ## add space between plots
plt.tight_layout(h_pad = 1)

fig, axs = plt.subplots(3, 1, figsize=(9, 4.5 * 3))
plt.suptitle(station_name+' during '+storm_name, size = 20)

    ## edit title and other labels/ limits
    ## Water Level
wl = axs[0].plot([], [])
axs[0].set_title('Water Level')
axs[0].set_xlabel('Date')
axs[0].set_ylabel('Water Level (ft above '+datum+')')
axs[0].set_ylim(ymax=max(df['Water Level']+1), ymin=min(df['Water Level'])-1)
axs[0].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[0].xaxis.set_major_formatter(date_form)

    ## Air Pressure
ap = axs[1].plot([], [])
axs[1].set_title('Air Pressure')
axs[1].set_xlabel('Date')
axs[1].set_ylabel('Air Pressure (mb)')
axs[1].set_ylim(ymax=max(df2['Air Pressure']+5), ymin=min(df2['Air Pressure'])-5)
axs[1].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[1].xaxis.set_major_formatter(date_form)

    ## Wind Speed and Wind Direction
wd = axs[2].plot([], [])
axs[2].quiver([], [], [], [], color = 'green', width = 0.0007, headwidth = 6.5)
# axs[2].quiver([], [], [], [])
axs[2].set_title('Wind Speed and Direction')
axs[2].set_xlabel('Date')
axs[2].set_ylabel('Wind Speed (knots)')
axs[2].set_ylim(ymax=max(df3['speed']+8), ymin=min(df3['speed'])-5)
axs[2].set_xlim(xmax=df.index[-1], xmin=df.index[0])
axs[2].xaxis.set_major_formatter(date_form)

    ## define lines to use in animation
lines = tuple(wl) + tuple(ap) + tuple(wd)

    ## define the start and end indices
a = df.index
b = df['Water Level']
c = df1['Air Pressure']
d = df3['speed']
e = df2['Air Temperature']
    ## update the lines in the subplot 
def update(frame):
    lines[0].set_data(a[:frame], b[:frame])
    lines[1].set_data(a[:frame], c[:frame])
    lines[2].set_data(a[:frame], d[:frame])
    lines[3].set_data(a[:frame], e[:frame])
    axs[2].quiver(a[:frame], d[:frame], U[:frame], V[:frame], color = 'black', width = 0.0007, headwidth = 6.5)
        # matplotlib does not currently have a function like set_data for quiver but this fix may be implemented soon
    return lines

    ##
def init():
    for line in lines:
        line.set_data([], [])
        lines[0].set_color('royalblue')
        lines[1].set_color('red')
        lines[2].set_color('green')
        lines[3].set_color('blue')
    return lines


    ## create the animation
ani = FuncAnimation(fig, update, frames=range(0,len(a),5), init_func=init, interval=100, blit=True)
    # frames can be updated to plot more or less data at once. Currently plots an additional 5 data points each consecutive frame


    ## add space between subplots
plt.tight_layout(h_pad = 1)

#%%time

#writer = FFMpegWriter(fps = 50) # fps = frames per second (the lower the slower)
#ani.save('CO-OPSDataAnimation.mp4', writer = writer)
#CPU times: total: 57.3 s
#Wall time: 57.6 s
print("end of request")
 
'''
