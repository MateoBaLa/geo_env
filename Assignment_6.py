#Install cdsapi and quary the ERA5 dataset of interest 

pip install "cdsapi>=0.7.4"

import cdsapi

dataset = "reanalysis-era5-single-levels"
request = {
    "product_type": ["reanalysis"],
    "variable": ["total_precipitation"],
    "year": [
        "2019", "2020", "2021",
        "2022", "2023", "2024"
    ],
    "month": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12"
    ],
    "day": [
        "01", "02", "03",
        "04", "05", "06",
        "07", "08", "09",
        "10", "11", "12",
        "13", "14", "15",
        "16", "17", "18",
        "19", "20", "21",
        "22", "23", "24",
        "25", "26", "27",
        "28", "29", "30",
        "31"
    ],
    "time": [
        "00:00", "01:00", "02:00",
        "03:00", "04:00", "05:00",
        "06:00", "07:00", "08:00",
        "09:00", "10:00", "11:00",
        "12:00", "13:00", "14:00",
        "15:00", "16:00", "17:00",
        "18:00", "19:00", "20:00",
        "21:00", "22:00", "23:00"
    ],
    "data_format": "netcdf",
    "download_format": "unarchived",
    "area": [23, 39, 22, 40]
}

client = cdsapi.Client()
client.retrieve(dataset, request).download()

#Import the necesary libraries 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

#Open the dataset 
dset = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\era5.nc')

#Extract the relevant variables from the dataset, including air temperature (t2m), precipitation (tp), latitude, longitude, and time. 
#convert these variables into numpy arrays
for further processing:
t2m = np.array(dset.variables['t2m'])
tp = np.array(dset.variables['tp'])  
latitude = np.array(dset.variables['latitude']) 
longitude = np.array(dset.variables['longitude'])
time_dt = np.array(dset.variables['time'])    

#Convert the air temperature from K to ◦C 
t2m = t2m - 273.15
tp = tp * 1000

#Check the dimension of the dataset and compute the mean across the second dimension to simplify the dataset
t2m.ndim
if t2m.ndim == 4:
    t2m = np.nanmean(t2m, axis=1)
    tp = np.nanmean(tp, axis=1)

#Create a Pandas dataframe containing time series data for air temperature and precipitation
  
df_era5 = pd.DataFrame(index= time_dt)
df_era5['t2m'] = t2m[:,3,2]
df_era5['tp'] = tp[:,3,2]

#plot the time series with an axis for precipitation and another for temperature 

fig, ax1 = plt.subplots(figsize=(12, 6))
plt.grid(True)
ax1.plot(df_era5.index, df_era5['t2m'], label='Temperature (◦C)', color='red', marker='x')
ax1.set_ylabel('2 meters Temperature (◦C)')
ax1.tick_params(axis='y', labelcolor='red')
ax1.legend(loc='upper left')
ax2 = ax1.twinx()
ax2.plot(df_era5.index, df_era5['tp'], label='Total Precipitation (mm/h)', color='blue', marker='o')
ax2.set_xlabel('Time')
ax2.set_ylabel('Precipitation (mm/h)')
ax2.tick_params(axis='y', labelcolor='blue')
ax2.legend(loc='upper right')
plt.title('Precipitation and 2 meters Temperature, Jeddah 2018 -2023 from ERA5')

plt.savefig('ERA5.png', dpi=300, bbox_inches='tight')
plt.show()

#Compute anual mean precipitation and temperature 
annual_precip = df_era5['tp'].resample('YE').mean()*24*365.25
mean_annual_precip = np.nanmean(annual_precip)

annual_tem = df_era5['t2m'].resample('YE').mean()#*24*365.25
mean_annual_tem = np.nanmean(annual_tem)


#Import tools to compute the Potential Evaporatiion using Hargreaves and Samani (1985) method

import tools
tmin = df_era5['t2m'].resample('D').min().values
tmax = df_era5['t2m'].resample('D').max().values
tmean = df_era5['t2m'].resample('D').mean().values
lat = 21.25
doy = df_era5['t2m'].resample('D').mean().index.dayofyear
pe = tools.hargreaves_samani_1982(tmin, tmax, tmean, lat, doy)

#Plot the Potential Evaporation time series 

ts_index = df_era5['t2m'].resample('D').mean().index
plt.figure()
plt.plot(ts_index, pe, label='Potential Evaporation')
plt.xlabel('Time')
plt.ylabel('Potential evaporation (mm d−1)')
plt.show()

#Create a Pandas dataframe containing time series data for potential evaporation and calculate anual mean 
pe_df = pd.DataFrame(index=ts_index)
pe_df['potential evaporation'] = pe 
annual_pe = pe_df['pe'].resample('YE').mean()*24*365.25
mean_annual_pep = np.nanmean(annual_pe)

#Calculate the volume of water loss for evaporation over a determined surface
#PE in meters/year
PE = mean_annual_pep/1000
#surface area in m^2
A = 1.6 * 10e6
#Volume in cubic meters 
V = A*PE