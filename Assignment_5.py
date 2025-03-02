#Import the necessary libraries to read the climate data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

#Open the datasets Data from  NCEI website specificly for November 25, 2009 from 00:00 UTC to 12:00 UTC
dset_06 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Jeddah_2009_Rainfall\GRIDSAT-B1.2009.11.25.06.v02r01.nc')
dset_00 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Jeddah_2009_Rainfall\GRIDSAT-B1.2009.11.25.00.v02r01.nc')
dset_03 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Jeddah_2009_Rainfall\GRIDSAT-B1.2009.11.25.03.v02r01.nc')
dset_09 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Jeddah_2009_Rainfall\GRIDSAT-B1.2009.11.25.09.v02r01.nc')
dset_12 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Jeddah_2009_Rainfall\GRIDSAT-B1.2009.11.25.12.v02r01.nc')

#Explore the dataset
IR = np.array(dset_06.variables['irwin_cdr']).squeeze()
IR.shape
IR = np.flipud(IR)
IR_C = IR*0.01+200
IR_C = IR_C-273.15
IR_C

#Plot a map to understand data, locate Jeddah 
plt.figure(1)
plt.imshow(IR_C, extent=[-180.035, 180.035, -70.035, 70.035], aspect='auto')
cbar = plt.colorbar()
cbar.set_label('Brightness temperature (degrees Celsius)')
jeddah_lat = 21.5
jeddah_lon = 39.2
plt.scatter(jeddah_lon, jeddah_lat, color='red', marker="o", label='Jeddah')
plt.savefig('Bright_Temp_C.png', dpi=300, bbox_inches='tight')

#Select the band of interest "irwin_cdr", and location of interest "Jeddah"
Jeddah_06 = dset_06['irwin_cdr'].sel(lat= 21.5, lon=39.375, method = "nearest")
Jeddah_09 = dset_09['irwin_cdr'].sel(lat= 21.5, lon=39.375, method = "nearest")
Jeddah_03 = dset_03['irwin_cdr'].sel(lat= 21.5, lon=39.375, method = "nearest")
Jeddah_00 = dset_00['irwin_cdr'].sel(lat= 21.5, lon=39.375, method = "nearest")
Jeddah_12 = dset_12['irwin_cdr'].sel(lat= 21.5, lon=39.375, method = "nearest")

#Merge the dataset and explore data 
Jeddah = xr.merge([Jeddah_12, Jeddah_09, Jeddah_06, Jeddah_03, Jeddah_00])
plt.scatter(Jeddah['time'], Jeddah["irwin_cdr"])
plt.show()

#Convert to dataframe to handle easily the variable of interest and made operations
JeddahRR = Jeddah.to_dataframe()

#Create three new columns for Temperature in C, Rainfall rate and cumulative rainfall rate  
JeddahRR["Temp_C"] =  (JeddahRR["irwin_cdr"]*0.01+200) -273.15
JeddahRR['Rainfall_Rate mm h^-1'] = 1.1183e11 * (2.718**(-3.6382e-2*(Jeddah["irwin_cdr"]**1.2)))
JeddahRR['Cumulative Rainfall mm h^-1'] = JeddahRR['Rainfall_Rate mm h^-1'].cumsum()

# Create the plot
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Rainfall Rate and Cumulative Rainfall on the first y-axis
ax1.plot(JeddahRR.index, JeddahRR['Rainfall_Rate mm h^-1'], label='Rainfall Rate (mm/h)', color='blue', marker='o')
ax1.plot(JeddahRR.index, JeddahRR['Cumulative Rainfall mm h^-1'], label='Cumulative Rainfall (mm)', color='green', marker='s')
ax1.set_xlabel('Time')
ax1.set_ylabel('Rainfall (mm)')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.legend(loc='upper left')

# Create a second y-axis for irwin_cdr
ax2 = ax1.twinx()
ax2.plot(JeddahRR.index, JeddahRR['irwin_cdr'], label='Temperature (K)', color='red', marker='x')
ax2.set_ylabel('Brightness Temperature (K)')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(loc='upper right')

# Add a title and grid
plt.title('Rainfall and Brightness Temperature, Jeddah November 25, 2009')
plt.grid(True)
plt.savefig('Jeddah_2009_Rainfall.png', dpi=300, bbox_inches='tight')
plt.show()