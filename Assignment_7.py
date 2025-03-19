import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
from netCDF4 import Dataset
import rioxarray as rio
import geopandas
from shapely.geometry import mapping

#Explore ERA5 kay variables, replace the path with the path to the different climatic datasets  
nc_file = Dataset(r"C:\Users\BARCOLM\geo_env\ERA5_datasets\Precipitation\era5_OLR_2000_total_precipitation.nc", "r")
print(nc_file.variables.keys())
nc_file.close()

#Impot Saudi Arabia boundaries shapefile
SaudiArabia_boundaries = r'C:\Users\BARCOLM\geo_env\Saudi_Shape_File\Saudi_Shape.shp'
SaudiArabia = geopandas.read_file(SaudiArabia_boundaries, crs="epsg:4326")

#Empty list of total precipitation monthly averages for months and years
all_monthly_mean = []
all_year_mean = []

#year range
start_year = 2000
end_year = 2020

#loop
for year in range (start_year, end_year +1):
    #file path for current year
    nc_fp = rf"C:\Users\BARCOLM\geo_env\ERA5_datasets\Precipitation\era5_OLR_{year}_total_precipitation.nc"
    #open dataset
    Rainfall = xr.open_dataset(nc_fp)
    #process the dataset
    Rainfall_clipped = Rainfall.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude')
    Rainfall_clipped = Rainfall_clipped.rio.write_crs("EPSG:4326")
    Rainfall_clipped = Rainfall_clipped.rio.clip(SaudiArabia.geometry.apply(mapping),drop=True)
    #calculate montlhy total rainfall
    Rainfall_monthly = Rainfall_clipped.groupby('valid_time.month').sum('valid_time')
    #calculate monthly average for the current year
    Rainfall_monthly_mean = Rainfall_monthly.mean(dim=('latitude', 'longitude')) * 1000  # average over space and Convert to mm
    all_monthly_mean.append(Rainfall_monthly_mean)
   
    #calculate year total rainfall
    Rainfall_yearly = Rainfall_clipped.groupby('valid_time.year').sum('valid_time')
    #calculate yearly average for the current year
    Rainfall_yearly_mean = Rainfall_yearly.mean(dim=('latitude', 'longitude')) * 1000  # average over space and Convert to mm
    all_year_mean.append(Rainfall_yearly_mean)

#Merge all the rainfall dataset for monthly and yearly averages   
combined_monthly_averages = xr.concat(all_monthly_mean, dim='year')
combined_yearly_averages = xr.concat(all_year_mean, dim='year')

#create the Data Frames  for monthly and yearly averages
monthly_avg_df = combined_monthly_averages.to_dataframe()
yearly_avg_df = combined_yearly_averages.to_dataframe()

#Organice yearly dataframe: reset index and define date time index 
yearly_avg_df.reset_index(inplace=True)
yearly_avg_df['Date'] = pd.to_datetime(yearly_avg_df['year'].astype(str))

#Organice Monthly dataframe: reset index and define date time index 
monthly_avg_df.reset_index(inplace=True)
monthly_avg_df.set_index(['year', 'month'], inplace=True)
monthly_avg_df.reset_index(inplace=True)  # Reset index to modify 'year' column
monthly_avg_df['year'] = monthly_avg_df['year'] + 2000  # Convert years to 2000-2020
monthly_avg_df['Date'] = pd.to_datetime(monthly_avg_df['year'].astype(str) + '-' + monthly_avg_df['month'].astype(str).str.zfill(2), format='%Y-%m')

# Plot Rainfall Monthly and Yearly averages the time series fro Saudi Arabia
# Create a figure and axis
plt.figure(figsize=(20, 6))
# Plot Monthly Averages
plt.plot(monthly_avg_df['Date'], monthly_avg_df['tp'], marker='o', linestyle='-', color='b', label='Monthly Average')
# Plot Yearly Averages
plt.plot(yearly_avg_df['Date'], yearly_avg_df['tp'], marker='s', linestyle='--', color='r', label='Yearly Average')
# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Rainfall (mm)')
plt.title('Monthly and Yearly Average Rainfall 2000 - 2020')
plt.legend()
plt.xticks(yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('Rainfall2000-2020.png', dpi=300, bbox_inches='tight')

#List of monthly and yearly averages and loop over dataset for total evaporation
evp_all_monthly_mean = []
evp_all_year_mean = []
for year in range (start_year, end_year +1):
    #file path for current year
    nc_fp_evp = rf"C:\Users\BARCOLM\geo_env\ERA5_datasets\Total_Evaporation\era5_OLR_{year}_total_evaporation.nc"
    #open dataset
    evp = xr.open_dataset(nc_fp_evp)
    #process the dataset
    evp_clipped = evp.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude')
    evp_clipped = evp_clipped.rio.write_crs("EPSG:4326")
    evp_clipped = evp_clipped.rio.clip(SaudiArabia.geometry.apply(mapping),drop=True)
    #calculate montlhy total evaporation 
    evp_monthly = evp_clipped.groupby('valid_time.month').sum('valid_time')
    #calculate monthly average for the current year
    evp_monthly_mean = evp_monthly.mean(dim=('latitude', 'longitude'))*(-1000)  # average over space and Convert to mm
    evp_all_monthly_mean.append(evp_monthly_mean)
   
    #calculate year total evaporation
    evp_yearly = evp_clipped.groupby('valid_time.year').sum('valid_time')
    #calculate yearly average for the current year
    evp_yearly_mean = evp_yearly.mean(dim=('latitude', 'longitude'))*(-1000)  # average over space and Convert to mm
    evp_all_year_mean.append(evp_yearly_mean)

#Merge datasets 
evp_combined_yearly_averages = xr.concat(evp_all_year_mean, dim='year')
evp_combined_monthly_averages = xr.concat(evp_all_monthly_mean, dim='year')

#Create and organice Data Frames for yearly and Monthly averages total evaporation
evp_monthly_avg_df = evp_combined_monthly_averages.to_dataframe()
evp_yearly_avg_df = evp_combined_yearly_averages.to_dataframe()
evp_yearly_avg_df.reset_index(inplace=True)
evp_yearly_avg_df['Date'] = pd.to_datetime(evp_yearly_avg_df['year'].astype(str))

evp_monthly_avg_df.reset_index(inplace=True)
evp_monthly_avg_df.set_index(['year', 'month'], inplace=True)
evp_monthly_avg_df.reset_index(inplace=True)  # Reset index to modify 'year' column
evp_monthly_avg_df['year'] = evp_monthly_avg_df['year'] + 2000  # Convert years to 2000-2020
evp_monthly_avg_df['Date'] = pd.to_datetime(evp_monthly_avg_df['year'].astype(str) + '-' + evp_monthly_avg_df['month'].astype(str).str.zfill(2), format='%Y-%m')

#Plot the time series for total evaporation monthly and yearly averages
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot Monthly Averages
plt.plot(evp_monthly_avg_df['Date'], evp_monthly_avg_df['e'], marker='o', linestyle='-', color='b', label='Monthly Average')

# Plot Yearly Averages
plt.plot(evp_yearly_avg_df['Date'], evp_yearly_avg_df['e'], marker='s', linestyle='--', color='r', label='Yearly Average')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Evaporation (mm water equivalent)')
plt.title('Monthly and Yearly Average Evaporation 2000 - 2020')
plt.legend()
plt.xticks(evp_yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('evaporation.png', dpi=300, bbox_inches='tight')

#Empty List of monthly and yearly averages for all years for runoff, and loop to over the dataset
ro_all_monthly_mean = []
ro_all_year_mean = []
for year in range (start_year, end_year +1):
    #file path for current year
    nc_fp_evp = rf"C:\Users\BARCOLM\geo_env\ERA5_datasets\Runoff\ambientera5_OLR_{year}_total_runoff.nc"
    #open dataset
    ro = xr.open_dataset(nc_fp_evp)
    #process the dataset
    ro_clipped = ro.rio.set_spatial_dims(x_dim='longitude',y_dim='latitude')
    ro_clipped = ro_clipped.rio.write_crs("EPSG:4326")
    ro_clipped = ro_clipped.rio.clip(SaudiArabia.geometry.apply(mapping),drop=True)
    #calculate montlhy total rainfall
    ro_monthly = ro_clipped.groupby('valid_time.month').sum('valid_time')
    #calculate monthly average for the current year
    ro_monthly_mean = ro_monthly.mean(dim=('latitude', 'longitude'))*1000  # average over space and Convert to mm
    ro_all_monthly_mean.append(ro_monthly_mean)
   
    #calculate year total rainfall
    ro_yearly = ro_clipped.groupby('valid_time.year').sum('valid_time')
    #calculate yearly average for the current year
    ro_yearly_mean = ro_yearly.mean(dim=('latitude', 'longitude'))*1000  # average over space and Convert to mm
    ro_all_year_mean.append(ro_yearly_mean)

#Merge the runoff datasets
ro_combined_yearly_averages = xr.concat(ro_all_year_mean, dim='year')
ro_combined_monthly_averages = xr.concat(ro_all_monthly_mean, dim='year')

#Create and organice Data Frames for yearly and monthly runoff averages 
ro_monthly_avg_df = ro_combined_monthly_averages.to_dataframe()
ro_yearly_avg_df = ro_combined_yearly_averages.to_dataframe()
ro_yearly_avg_df.reset_index(inplace=True)
ro_yearly_avg_df['Date'] = pd.to_datetime(ro_yearly_avg_df['year'].astype(str))

ro_monthly_avg_df.reset_index(inplace=True)
ro_monthly_avg_df.set_index(['year', 'month'], inplace=True)
ro_monthly_avg_df.reset_index(inplace=True)  # Reset index to modify 'year' column
ro_monthly_avg_df['year'] = ro_monthly_avg_df['year'] + 2000  # Convert years to 2000-2020
ro_monthly_avg_df['Date'] = pd.to_datetime(ro_monthly_avg_df['year'].astype(str) + '-' + ro_monthly_avg_df['month'].astype(str).str.zfill(2), format='%Y-%m')

#Plot the time series for yearly and monthy average runoff 
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot Monthly Averages
plt.plot(ro_monthly_avg_df['Date'], ro_monthly_avg_df['ro'], marker='o', linestyle='-', color='b', label='Monthly Average')

# Plot Yearly Averages
plt.plot(ro_yearly_avg_df['Date'], ro_yearly_avg_df['ro'], marker='s', linestyle='--', color='r', label='Yearly Average')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Runoff (mm)')
plt.title('Monthly and Yearly Average Runoff 2000 - 2020')
plt.legend()
plt.xticks(evp_yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('Runoff.png', dpi=300, bbox_inches='tight')

#Create monthly and yearly WATER BALANCE Data Frame and calculate it using: precipitation - (evaporation + runoff)
water_balance_df = pd.DataFrame()
water_balance_df['wb'] = monthly_avg_df['tp'] - (evp_monthly_avg_df['e'] + ro_monthly_avg_df['ro'])
water_balance_df["Date"] = monthly_avg_df['Date']

water_balance_df['evp-ro'] = (evp_monthly_avg_df['e'] + ro_monthly_avg_df['ro'])

yearly_water_balance_df = pd.DataFrame()
yearly_water_balance_df['wb'] = yearly_avg_df['tp'] - (evp_yearly_avg_df['e'] + ro_yearly_avg_df['ro'])
yearly_water_balance_df["Date"] = yearly_avg_df['Date']
yearly_water_balance_df

#Create time series figure for WATER BALANCE
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot Monthly Averages
plt.plot(water_balance_df['Date'], water_balance_df['wb'], marker='o', linestyle='-', color='b', label='Monthly Average')

# Plot Yearly Averages
plt.plot(yearly_water_balance_df['Date'], yearly_water_balance_df['wb'], marker='s', linestyle='--', color='r', label='Yearly Average')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Water Balance mm')
plt.title('Monthly and Yearly Water Balance  2000 - 2020')
plt.legend()
plt.xticks(evp_yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('WB.png', dpi=300, bbox_inches='tight')

#Create time series figure for rainfall and evaporation values 
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot Monthly Averages
plt.plot(monthly_avg_df['Date'], monthly_avg_df['tp'], marker='o', linestyle='-', color='b', label='Rainfall Monthly Average')

# Plot Yearly Averages
plt.plot(evp_monthly_avg_df['Date'], evp_monthly_avg_df['e'], marker='s', linestyle='--', color='r', label='Evaporation Monthly Average')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Rainfall - Evaporation (mm)')
plt.title('Monthly Average Evaporation and Rainfall 2000 - 2020')
plt.legend()
plt.xticks(evp_yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('evap_tp.png', dpi=300, bbox_inches='tight')

#Create a two Y axis figure to compare trends between yearly runoff and rainfall averages
# Create a figure and axis
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot Monthly Averages
ax1.plot(yearly_avg_df['Date'], yearly_avg_df['tp'], marker='o', linestyle='-', color='b', label='Yearly Average rainfall')
ax1.set_xlabel('Year')
ax1.set_ylabel('Yearly Average Rainfall (mm)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
#ax1.set_xticks(range(2000, 2020))  # Set x-ticks for months (1 to 12)
#ax1.set.xticks(yearly_avg_df['Date'])
ax1.grid(True)

# Create a second y-axis for Yearly Averages
ax2 = ax1.twinx()
ax2.plot(ro_yearly_avg_df['Date'], ro_yearly_avg_df['ro'], marker='s', linestyle='--', color='r', label='Yearly Average runoff')
ax2.set_ylabel('Yearly Average Runoff (mm)', color='r')
ax2.tick_params(axis='y', labelcolor='r')

# Add a title and legend
 
plt.title('Yearly Average Rainfall and Runoff')
fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.9))

# Show the plot

plt.tight_layout()
plt.savefig('Rain_Runoff.png', dpi=300, bbox_inches='tight')
plt.show()

#Create Time Series figure to compare evaration plus runoff and evaporation with monthly averages data frames
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot Monthly Averages
plt.plot( water_balance_df['Date'], water_balance_df['evp-ro'], marker='o', linestyle='-', color='b', label='Evaporation plus runoff ')

# Plot Yearly Averages
plt.plot(evp_monthly_avg_df['Date'], evp_monthly_avg_df['e'], marker='s', linestyle='--', color='r', label='Evaporation Monthly Average')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('mm')
plt.title('Average Evaporation and Evaporation plus Runoff 2000 - 2020')
plt.legend()
plt.xticks(evp_yearly_avg_df['Date'], rotation=45)  # Set x-ticks to each year and rotate labels
plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
plt.grid(True)
plt.savefig('evap_ept_ro.png', dpi=300, bbox_inches='tight')