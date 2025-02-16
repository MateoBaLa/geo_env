#Import the necessary libraries to read the climate data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
#Import the tool to calculate Heat Index
import tools

#Open the folder with the ISD data for Jeddah airport - change the path to your local folder-
df_isd = tools.read_isd_csv(r'C:\Users\BARCOLM\geo_env\41024099999.csv')

#Show the figure for the ISD data from Jeddah airport 
plt.figure()
df_isd.plot(title= "ISD data for Jeddah", cmap='viridis')
plt.savefig('ISD_Jeddah.png', dpi=300, bbox_inches='tight')
plt.show()

#Calculate the Humid Index using the imported tools
df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW'].values, df_isd['TMP'].values)
df_isd['HI'] = tools.gen_heat_index(df_isd['TMP'].values, df_isd['RH'].values)

#Explore data
df_isd.max()
df_isd.min()
df_isd.idxmax()

#HI local time 2pm Aug 10 - 2024
df_isd.loc["2024-08-10 11:00:00"]

#Calculate the HI using daily weather instead of hourly... use resample pandas df method. And explore the data with a figure
resample_df = df_isd.resample('D').mean()
resample_df
plt.figure()
resample_df['HI'].plot()
plt.savefig('HI_Daily_v1.png', dpi=300, bbox_inches='tight')
plt.show()

#Open the folder with the SSP245 data and calculate the mean 
dset_245 = xr.open_dataset(r'C:\Users\BARCOLM\Documents\geo_env_v2\tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc')


#Exploring Jeddah coordinates 
dset_245['lat'].values
dset_245['lon'].values

#We query SSP245 for the specitfic coordinates and time (2071-2100) for Jeddah 
Jeddah_SSP245 = dset_245.sel(lat= 21.5, lon=39.375)
Jeddah_245 = Jeddah_SSP245['tas'].sel(time=slice('20710101','21001231'))

#Convert to dataframe 
Jeddah = Jeddah_245.to_dataframe()

#Organice the dataframe
Jeddah_df = Jeddah.rename(columns={"tas": "Temp_K"})
Jeddah_df["Temp_C"] =  Jeddah_df["Temp_K"]-273.15

#Calculate delta between observed period (2024) and SSP245
Jeddah_spp245 = Jeddah_df["Temp_C"].mean()
df_isd_mean = df_isd["TMP"].mean()
Delta = df_isd_mean - Jeddah_spp245

#Calculate the projected temperature for the SSP245 scenario 
df_isd["DEW_SSP245"] = df_isd["DEW"]+Delta
df_isd["TEM_SSP245"] = df_isd["TMP"]+Delta

#Calculate Relative Humidity and Heat Index with the projected incresed Temperature 
df_isd['RH'] = tools.dewpoint_to_rh(df_isd['DEW_SSP245'].values, df_isd['TMP_SSP245'].values)
df_isd['HI'] = tools.gen_heat_index(df_isd['TMP_SSP245'].values, df_isd['RH'].values)

#Explore the change in max heat index
df_isd.max()