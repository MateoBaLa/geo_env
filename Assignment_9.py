#conda install dask --force-reinstall
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr
from netCDF4 import Dataset
import rioxarray as rio
import geopandas as gpd
from shapely.geometry import mapping
import scipy.optimize as opt
import os
from scipy.stats import norm
from collections import namedtuple

#Combine datasets 
# Path to your directory
path = r'C:\Users\BARCOLM\geo_env\Temp_126\Temp_126'

# List all the files in the directory with the pattern
files = [
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2015_2020.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2021_2030.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2031_2040.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2041_2050.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2051_2060.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2061_2070.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2071_2080.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2081_2090.nc',
    'gfdl-esm4_r1i1p1f1_w5e5_ssp370_tas_lat15.0to33.0lon33.0to60.0_daily_2091_2100.nc'
]
# Create full paths
full_paths = [os.path.join(path, f) for f in files]

# Open and concatenate along time dimension
combined = xr.open_mfdataset(full_paths, combine='nested', concat_dim='time')

# Save to new file
output_path = os.path.join(path, 'Temp370.nc')
combined.to_netcdf(output_path)

print(f"Files successfully combined and saved to {output_path}")

# ----------------------------------------------------------
# TREND ANALYSIS addjusted Mann-Kendall test FUNCTIONS developed by  (Hamed & Rao 1998 + Sen's Slope) --
# ----------------------------------------------------------

def hamed_rao_mk_test(x, alpha=0.05):
    """Modified MK test with autocorrelation correction (Hamed & Rao 1998)"""
    n = len(x)
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])
    
    # Calculate variance with autocorrelation correction
    var_s = n*(n-1)*(2*n+5)/18
    ties = np.unique(x, return_counts=True)[1]
    for t in ties:
        var_s -= t*(t-1)*(2*t+5)/18
    
    # Correct for autocorrelation
    n_eff = n
    if n > 10:
        acf = [1] + [np.corrcoef(x[:-i], x[i:])[0,1] for i in range(1, n//4)]
        n_eff = n / (1 + 2 * sum((n-i)/n * acf[i] for i in range(1, len(acf))))
        var_s *= n_eff / n
    
    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0
    
    p = 2 * (1 - norm.cdf(abs(z)))
    h = abs(z) > norm.ppf(1-alpha/2)
    
    Trend = namedtuple('Trend', ['trend', 'h', 'p', 'z', 's'])
    trend = 'increasing' if s > 0 else 'decreasing' if s < 0 else 'no trend'
    return Trend(trend=trend, h=h, p=p, z=z, s=s)

def sens_slope(x, y):
    """Sen's slope estimator"""
    slopes = []
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    return np.median(slopes)


#Open de temperature combined datasets    
Temp370 = xr.open_dataset(r"C:\Users\BARCOLM\geo_env\Temp_370\Temp_370\Temp370.nc")
Temp126 = xr.open_dataset(r"C:\Users\BARCOLM\geo_env\Temp_126\Temp_126\Temp126.nc")

#Calculate yearly mean and reagrup by year in bot datasets SSP126 and SSP370
Temp_yearly = Temp370.groupby('time.year').mean('time')
#calculate yearly average 
Temp_yearly_mean = Temp_yearly.mean(dim=('lat', 'lon'))  # average over space and Convert 

Temp_yearly_126 = Temp126.groupby('time.year').mean('time')
#calculate yearly average
Temp_yearly_mean_126 = Temp_yearly_126.mean(dim=('lat', 'lon'))  # average over space 

#Create the dataframe for temperature under both scnarios 
T_df = pd.DataFrame({'Year': Temp_yearly_mean['year'], 'Temp370': Temp_yearly_mean['tas'], 
'Temp126': Temp_yearly_mean_126['tas']})

# Create a figure and axis to plot the temperature over time in both scenarios 
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(T_df['Year'], T_df['Temp370'], marker='o', linestyle='--', color='b', label='Temp SSP3- RCP7.0')
plt.plot(T_df['Year'], T_df['Temp126'], marker='s', linestyle='-', color='r', label='Temp SSP1- RCP2.6')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('Temperature (K)')
plt.title('Yearly mean Temperature (K) SSP1- RCP2.6 & SSP3- RCP7.0 scenarios')
plt.legend()

plt.savefig('Temp_mean_spp126-370.png', dpi=300, bbox_inches='tight')

#Aply the function to calculate and report Mann Kendal test and Sen's slope 
Temp370_Trend = hamed_rao_mk_test(T_df['Temp370'], alpha=0.05)
Temp370_slope = sens_slope(T_df['Year'], T_df['Temp370'])
Temp126_Trend = hamed_rao_mk_test(T_df['Temp126'], alpha=0.05)
Temp126_slope = sens_slope(T_df['Year'], T_df['Temp126'])

#Open de precipitation combined datasets
rain126 = xr.open_dataset(r'C:\Users\BARCOLM\geo_env\Precipitation_SSP370_126\PR_126.nc')
rain370 = xr.open_dataset(r"C:\Users\BARCOLM\geo_env\Precipitation_SSP370_126\Precipitation_370\pr370.nc")

#Calculate yearly mean and reagrup by year in bot datasets SSP126 and SSP370
rain126_yearly = rain126.groupby('time.year').sum('time')
#calculate yearly average 
rain126_yearly_mean = rain126_yearly.mean(dim=('lat', 'lon'))  # average over space and Convert to 
rain370_yearly = rain370.groupby('time.year').sum('time')
#calculate yearly average for the current year
rain370_yearly_mean = rain370_yearly.mean(dim=('lat', 'lon'))  # average over space and Convert to 

##Create the dataframe for precipitationunder both scenarios
rain_df = pd.DataFrame({'Year': rain126_yearly_mean['year'], 'rain370': rain370_yearly_mean['pr'], 
'rain126': rain126_yearly_mean['pr']})

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(rain_df['Year'], rain_df['rain370'], marker='o', linestyle='--', color='b', label='rain SSP3- RCP7.0 ')
plt.plot(T_df['Year'], rain_df['rain126'], marker='s', linestyle='-', color='r', label='rain SSP1- RCP2.6')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('precipitation flux (kg m-2 s-1)')
plt.title('Precipitation vs Years under the SSP1- RCP2.6, and SSP3- RCP7.0 scenarios.')
plt.legend()
plt.savefig('rain_mean_spp126-370.png', dpi=300, bbox_inches='tight')

rain370_Trend = hamed_rao_mk_test(rain_df['rain370'], alpha=0.05)
rain370_slope = sens_slope(rain_df['Year'], rain_df['rain370'])

print(f"Trend detected: {rain370_Trend}")
print(f"Sen's Slope: {rain370_slope} ")
rain126_Trend = hamed_rao_mk_test(rain_df['rain126'], alpha=0.05)
rain126_slope = sens_slope(rain_df['Year'], rain_df['rain126'])

print(f"Trend detected: {rain126_Trend}")
print(f"Sen's Slope: {rain126_slope} ")

#Calculate max temperature, plot,report trend slope  for both scenarios 

Temp_day_126 = Temp126.groupby('time.year').max('time')
#calculate max temperature
Temp_day_max_126 = Temp_day_126.max(dim=('lat', 'lon'))         

Temp_day_370 = Temp370.groupby('time.year').max('time')
#calculate max
Temp_day_max_370 = Temp_day_370.max(dim=('lat', 'lon'))  

T_df_max = pd.DataFrame({'Year': Temp_day_max_370['year'], 'Temp370_max': Temp_day_max_370['tas'],
 'Temp126_max': Temp_day_max_126['tas']})
 
 # Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(T_df_max['Year'], T_df_max['Temp370_max'], marker='o', linestyle='--', color='b', label='Max Temp SSP3- RCP7.0')
plt.plot(T_df_max['Year'], T_df_max['Temp126_max'], marker='s', linestyle='-', color='r', label='Max Temp SSP1- RCP2.6')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('Temperature (K)')
plt.title('Yearly max Temperature (K) SSP1- RCP2.6 & SSP3- RCP7.0 scenarios')
plt.legend()

plt.savefig('Temp_max_ssp126-370.png', dpi=300, bbox_inches='tight')

Temp370_max_Trend = hamed_rao_mk_test(T_df_max['Temp370_max'], alpha=0.05)
Temp370_max_slope = sens_slope(T_df_max['Year'], T_df_max['Temp370_max'])

print(f"Trend detected: {Temp370_max_Trend}")
print(f"Sen's Slope: {Temp370_max_slope} ")

Temp126_max_Trend = hamed_rao_mk_test(T_df_max['Temp126_max'], alpha=0.05)
Temp126_max_slope = sens_slope(T_df_max['Year'], T_df_max['Temp126_max'])

print(f"Trend detected: {Temp126_max_Trend}")
print(f"Sen's Slope: {Temp126_max_slope} ")

#Calculate max precipitationt trend slope for both scenarios 
rain126_max = rain126.groupby('time.year').sum('time')
#calculate yearly max for the current year
rain126_yearly_max = rain126_max.max(dim=('lat', 'lon'))

rain370_max = rain370.groupby('time.year').sum('time')
#calculate yearly max precipitation for the current year
rain370_yearly_max = rain370_max.max(dim=('lat', 'lon'))

rain_df_max = pd.DataFrame({'Year': rain126_yearly_max['year'], 'rain370_max': rain370_yearly_max['pr'], 
'rain126_max': rain126_yearly_max['pr']})

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(rain_df_max['Year'], rain_df_max['rain370_max'], marker='o', linestyle='--', color='b', label='max rain SSP3- RCP7.0')
plt.plot(rain_df_max['Year'], rain_df_max['rain126_max'], marker='s', linestyle='-', color='r', label='max rain SSP1- RCP2.6')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('max precipitation flux')
plt.title('Maximum Precipitation vs Years under the SSP1- RCP2.6, and SSP3- RCP7.0 scenarios.')
plt.legend()

plt.savefig('rain_max_ssp126-370', dpi=300, bbox_inches='tight')

rain370_Trend_max = hamed_rao_mk_test(rain_df_max['rain370_max'], alpha=0.05)
rain370_slope_max = sens_slope(rain_df_max['Year'], rain_df_max['rain370_max'])

print(f"Trend detected: {rain370_Trend_max}")
print(f"Sen's Slope: {rain370_slope_max} ")

rain126_Trend_max = hamed_rao_mk_test(rain_df_max['rain126_max'], alpha=0.05)
rain126_slope_max = sens_slope(rain_df_max['Year'], rain_df_max['rain126_max'])

print(f"Trend detected: {rain126_Trend_max}")
print(f"Sen's Slope: {rain126_slope_max} ")


###Calculate wet bulb

def calculate_wet_bulb_temperature(temp_k, rh_percent):
    """
    Calculate wet bulb temperature from air temperature and relative humidity.
    
    Args:
        temp_k: Temperature in Kelvin
        rh_percent: Relative humidity in percent
        
    Returns:
        Wet bulb temperature in Kelvin
    """
    # Convert temperature from Kelvin to Celsius for calculations
    temp_c = temp_k - 273.15
    
    # Calculation using Stull's method (2011) - accurate to within 0.3°C
    wbt_c = temp_c * np.arctan(0.151977 * (rh_percent + 8.313659)**0.5) + \
            np.arctan(temp_c + rh_percent) - np.arctan(rh_percent - 1.676331) + \
            0.00391838 * (rh_percent)**(3/2) * np.arctan(0.023101 * rh_percent) - 4.686035
    
    # Convert back to Kelvin
    wbt_k = wbt_c + 273.15
    
    return wbt_k
def main():
    # Input file paths
    temp_file = r"C:\Users\BARCOLM\geo_env\Temp_370\Temp_370\Temp370.nc"
    rh_file = r"C:\Users\BARCOLM\geo_env\Humidity_370_126\Humudity_370\RH370.nc"
    
    # Output directory and file
    output_dir = r"C:\Users\BARCOLM\geo_env\wet_bulb_temp"
    output_file = os.path.join(output_dir, "wb_370.nc")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the datasets
    ds_temp = xr.open_dataset(temp_file)
    ds_rh = xr.open_dataset(rh_file)
    
    # Extract temperature and humidity data
    temp_k = ds_temp['tas']  # Assuming 'tas' is temperature variable
    rh_percent = ds_rh['hurs']  # Assuming 'hurs' is relative humidity
    
    # Calculate wet bulb temperature
    wbt_k = calculate_wet_bulb_temperature(temp_k, rh_percent)
    
    # Create a new dataset for the output
    ds_output = xr.Dataset(
        {
            'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_k.values),
        },
        coords={
            'time': ds_temp['time'],
            'lat': ds_temp['lat'],
            'lon': ds_temp['lon'],
        },
        attrs={
            'description': 'Wet bulb temperature calculated from temperature and relative humidity',
            'units': 'K',
            'calculation_method': "Stull's method (2011)",
        }
    )
    
    # Save to NetCDF
    ds_output.to_netcdf(output_file)
    print(f"Wet bulb temperature saved to: {output_file}")
    
    # Close the datasets
    ds_temp.close()
    ds_rh.close()

if __name__ == "__main__":
    main()
    
def main():
    # Input file paths
    temp_file = r"C:\Users\BARCOLM\geo_env\Temp_126\Temp_126\Temp126.nc"
    rh_file = r"C:\Users\BARCOLM\geo_env\Humidity_370_126\RH126.nc"
    
    # Output directory and file
    output_dir = r"C:\Users\BARCOLM\geo_env\wet_bulb_temp"
    output_file = os.path.join(output_dir, "wb_126.nc")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Load the datasets
    ds_temp = xr.open_dataset(temp_file)
    ds_rh = xr.open_dataset(rh_file)
    
    # Extract temperature and humidity data
    temp_k = ds_temp['tas']  # Assuming 'tas' is temperature variable
    rh_percent = ds_rh['hurs']  # Assuming 'hurs' is relative humidity
    
    # Calculate wet bulb temperature
    wbt_k = calculate_wet_bulb_temperature(temp_k, rh_percent)
    
    # Create a new dataset for the output
    ds_output = xr.Dataset(
        {
            'wet_bulb_temp': (['time', 'lat', 'lon'], wbt_k.values),
        },
        coords={
            'time': ds_temp['time'],
            'lat': ds_temp['lat'],
            'lon': ds_temp['lon'],
        },
        attrs={
            'description': 'Wet bulb temperature calculated from temperature and relative humidity',
            'units': 'K',
            'calculation_method': "Stull's method (2011)",
        }
    )
    
    # Save to NetCDF
    ds_output.to_netcdf(output_file)
    print(f"Wet bulb temperature saved to: {output_file}")
    
    # Close the datasets
    ds_temp.close()
    ds_rh.close()

if __name__ == "__main__":
    main()
    
    
#Create dataset for wet bulb for both scenarios, mean and max. Report trend and slope
    
wb126 = xr.open_dataset(r"C:\Users\BARCOLM\geo_env\wet_bulb_temp\wb_126.nc")
wb370 = xr.open_dataset(r"C:\Users\BARCOLM\geo_env\wet_bulb_temp\wb_370.nc")


wb126_mean = wb126.groupby('time.year').mean('time')

wb126_yearly_mean = wb126_mean.mean(dim=('lat', 'lon'))

wb370_mean = wb370.groupby('time.year').mean('time')

wb370_yearly_mean = wb370_mean.mean(dim=('lat', 'lon'))                                    

wb_df = pd.DataFrame({'Year': wb370_yearly_mean['year'], 'wb370 mean': wb370_yearly_mean['wet_bulb_temp'],
 'wb126 mean': wb126_yearly_mean['wet_bulb_temp']})
 
 wb126_max = wb126.groupby('time.year').max('time')
#calculate yearly mmean for the current year
wb126_yearly_max = wb126_mean.max(dim=('lat', 'lon'))

wb370_max = wb370.groupby('time.year').max('time')
#calculate yearly for the current year
wb370_yearly_max = wb370_mean.max(dim=('lat', 'lon'))                                    

wb_df_max = pd.DataFrame({'Year': wb370_yearly_max['year'], 'wb370 max': wb370_yearly_max['wet_bulb_temp'], 
'wb126 max': wb126_yearly_max['wet_bulb_temp']})


# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(wb_df['Year'], wb_df['wb370 mean'], marker='o', linestyle='--', color='b', label='mean wet bulb temperature (k) ssp 370')
plt.plot(wb_df['Year'], wb_df['wb126 mean'], marker='s', linestyle='-', color='r', label='mean wet bulb temperature (k) ssp 126')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('mean wet bulb temperature (k)')
plt.title('Wet bulb temperature of Saudi across (2015-2100) under the SSP3- RCP7.0')
plt.legend()
#plt.xticks(df['Time'], rotation=45)  # Set x-ticks to each year and rotate labels
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
#plt.grid(True)
plt.savefig('wb_mean_ssp126-370', dpi=300, bbox_inches='tight')

wb370_mean_Trend = hamed_rao_mk_test(wb_df['wb370 mean'], alpha=0.05)
wb370_mean_slope = sens_slope(wb_df['Year'], wb_df['wb370 mean'])

print(f"Trend detected: wb370 {wb370_mean_Trend}")
print(f"Sen's Slope: wb370 {wb370_mean_slope} ")

wb126_mean_Trend = hamed_rao_mk_test(wb_df['wb126 mean'], alpha=0.05)
wb126_mean_slope = sens_slope(wb_df['Year'], wb_df['wb126 mean'])

print(f"Trend detected: wb 126 {wb126_mean_Trend}")
print(f"Sen's Slope: wb 126 {wb126_mean_slope} ")

#Sens slope function with intersection 

def sens_slope_with_intercept(x, y):
    """Sen's slope with intercept calculation"""
    x = np.asarray(x)
    y = np.asarray(y)
    n = len(x)
    slopes = []
    
    # Calculate pairwise slopes
    for i in range(n):
        for j in range(i+1, n):
            if x[j] != x[i]:
                slopes.append((y[j] - y[i]) / (x[j] - x[i]))
    
    slope = np.median(slopes)
    intercept = np.median(y - slope * x)  # Median of residuals
    return slope, intercept
    
wb126_mean_slope2, intercept126 = sens_slope_with_intercept(wb_df['Year'], wb_df['wb126 mean'])
wb370_mean_slope2, intercept370 = sens_slope_with_intercept(wb_df['Year'], wb_df['wb370 mean'])

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(wb_df['Year'], wb_df['wb126 mean'], label='mean wet bulb temperature (k) ssp 126')
plt.plot(wb_df['Year'], intercept126 + wb126_mean_slope2 * wb_df['Year'], 'r--', 
         label=f'Sen\'s Slope: {wb126_mean_slope2:.4f}/yr ')
plt.xlabel('Year')
plt.ylabel('mean wet bulb temperature (k) ssp 126')
plt.legend()
plt.grid(True)
plt.savefig('wb_scatter_ssp126', dpi=300, bbox_inches='tight')
plt.show()

# Plot
plt.figure(figsize=(10, 5))
plt.scatter(wb_df['Year'], wb_df['wb370 mean'], label='mean wet bulb temperature (k) ssp 370')
plt.plot(wb_df['Year'], intercept370 + wb370_mean_slope2 * wb_df['Year'], 'r--', 
         label=f'Sen\'s Slope: {wb370_mean_slope2:.4f}/yr ')
plt.xlabel('Year')
plt.ylabel('mean wet bulb temperature (k) ssp 370')
plt.legend()
plt.grid(True)
plt.savefig('wb_scatter_ssp370', dpi=300, bbox_inches='tight')
plt.show()

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(wb_df_max['Year'], wb_df_max['wb370 max'], marker='o', linestyle='--', color='b', label='max wet bulb temperature (k) ssp 370')
plt.plot(wb_df_max['Year'], wb_df_max['wb126 max'], marker='s', linestyle='-', color='r', label='max wet bulb temperature (k) ssp 126')

# Add labels, title, and legend
plt.xlabel('Year')
plt.ylabel('max wet bulb temperature (k)')
plt.title('Yearly max wet bulb temperature (k)  SSP 370 & SSP 126')
plt.legend()
#plt.xticks(df['Time'], rotation=45)  # Set x-ticks to each year and rotate labels
#plt.grid(True, which='both', linestyle='--', linewidth=0.5)  # Add grid lines
#plt.grid(True)
plt.savefig('wb_maxx_spp16-370', dpi=300, bbox_inches='tight')

wb370_max_Trend = hamed_rao_mk_test(wb_df_max['wb370 max'], alpha=0.05)
wb370_max_slope = sens_slope(wb_df_max['Year'], wb_df_max['wb370 max'])

print(f"Trend detected wb370_max: {wb370_max_Trend}")
print(f"Sen's Slope wb370_max: {wb370_max_slope} ")

wb126_max_Trend = hamed_rao_mk_test(wb_df_max['wb126 max'], alpha=0.05)
wb126_max_slope = sens_slope(wb_df_max['Year'], wb_df_max['wb126 max'])

print(f"Trend detected wb 126 max: wb 126 {wb126_max_Trend}")
print(f"Sen's Slope wb 126 max:  {wb126_max_slope} ")