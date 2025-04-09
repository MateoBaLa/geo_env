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

#os.chdir(os.path.abspath(''))
os.chdir(os.path.dirname(r'C:\Users\BARCOLM\geo_env\WaterShet'))
print(os.getcwd())
## ---Part 1: Pre-Processing ERA5 dataset ---

# Clip each variable using the shapefile
def load_and_clip(nc_file, var_name, gdf):
    ds = xr.open_dataset(nc_file)
    ds = ds.rio.write_crs("EPSG:4326")  # Ensure correct CRS
    clipped = ds.rio.clip(gdf.geometry, gdf.crs, drop=True)
    return clipped[var_name]

# Load the watershed shapefile
shapefile_path = r'C:\Users\BARCOLM\geo_env\WaterShet\WS_3.shp'
gdf = gpd.read_file(shapefile_path)

# Load the NetCDF files (precipitation, ET, runoff)
precip_file = r'C:\Users\BARCOLM\geo_env\WaterShet\era5_OLR_2001_total_precipitation.nc'
et_file = r'C:\Users\BARCOLM\geo_env\WaterShet\era5_OLR_2001_total_evaporation.nc'
runoff_file = r'C:\Users\BARCOLM\geo_env\WaterShet\ambientera5_OLR_2001_total_runoff.nc'

# precip_ds = xr.open_dataset(precip_file)
# et_ds = xr.open_dataset(et_file)

# ds_clipped = clip_to_shapefile(ds, gdf)

# Extract variables
# Load and clip each dataset,unit conversion: meters to mm
P_grid = load_and_clip(precip_file, "tp", gdf) * 1000.0
ET_grid = load_and_clip(et_file, "e", gdf) * 1000.0
Q_grid = load_and_clip(runoff_file, "ro", gdf) * 1000.0

# Compute area-averaged values
P = P_grid.mean(dim=["latitude", "longitude"]).values
ET = ET_grid.mean(dim=["latitude", "longitude"]).values
Q_obs = Q_grid.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET = np.where(ET < 0.0, -ET, ET) 

## --- Part 2: Model setup and calibration ---

# Rainfall-runoff model (finite difference approximation)
def simulate_runoff(k, P, ET, dt=1):
    n = len(P)
    Q_sim = np.zeros(n)
    Q_sim[0] = Q_obs[0]
    
    for t in range(2, n):
        Q_t = (Q_sim[t-1] + (P[t] - ET[t]) * dt) / (1 + dt/k)
        Q_sim[t] = max(0,Q_t) # Ensure non-negative runoff

    return (Q_sim)
    
# Define the objective (KGE) function
def kge(Q_obs, Q_sim):
    r = np.corrcoef(Q_obs, Q_sim)[0, 1]
    alpha = np.std(Q_sim) / np.std(Q_obs)
    beta = np.mean(Q_sim) / np.mean(Q_obs)
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    #print interative matrix, if needed
    #print(r, alpha, beta, kge)
    return (kge, r, alpha, beta)

# Create the list of k values and run the model to get simulated runoff and performance index
k_testlist = np.arange(0.15, 0.3, 0.15)
#k_testlist = 0.15
Q_sim_all = np.empty([len(P), len(k_testlist)])
PerfIndex_all = np.empty([len(k_testlist), 5]) #for k, kge, r, alpha, beta

n=0
for k in k_testlist:
    Qsim = simulate_runoff(k, P, ET)
    Q_sim_all[:,n] = Qsim
    
    PerfIndex = kge(Q_obs, Qsim)
    PerfIndex_all[n,0] = k
    PerfIndex_all[n,1:] = PerfIndex
    n += 1

#print (Q_sim_all)
print (PerfIndex_all)

# Objective function for optimization
def objective(k, P, ET, Q_obs):
    Q_sim = simulate_runoff(k, P, ET)
    kge_model = kge(Q_obs, Q_sim)
    return (1.0 - kge_model[0])

# Optimize k using KGE
res = opt.minimize_scalar(objective, bounds=(0.1, 2), args=(P, ET, Q_obs), method='bounded')
print(res)

# Best k value
best_k = res.x
Q_sim = simulate_runoff(best_k, P, ET)
print(f"Optimized k: {best_k:.3f}")

# After optimization 
best_k = res.x  

# Simulate runoff with the best k
Q_sim_calibrated = simulate_runoff(best_k, P, ET)

# Calculate KGE and its components
kge_calibrated, r, alpha, beta = kge(Q_obs, Q_sim_calibrated)

# Print all metrics
print(f"Calibration KGE: {kge_calibrated:.3f}")
print(f"Correlation (r): {r:.3f}")
print(f"Variability Ratio (alpha): {alpha:.3f}")
print(f"Bias Ratio (beta): {beta:.3f}")

# --- Validation ---

# Load the NetCDF files for validation (precipitation, ET, runoff)
precip_fileVal = r'C:\Users\BARCOLM\geo_env\WaterShet\era5_OLR_2002_total_precipitation.nc'
et_fileVal = r'C:\Users\BARCOLM\geo_env\WaterShet\era5_OLR_2002_total_evaporation.nc'
runoff_fileVal = r'C:\Users\BARCOLM\geo_env\WaterShet\ambientera5_OLR_2002_total_runoff.nc'

P_gridVal = load_and_clip(precip_fileVal, "tp", gdf) * 1000.0
ET_gridVal = load_and_clip(et_fileVal, "e", gdf) * 1000.0
Q_gridVal = load_and_clip(runoff_fileVal, "ro", gdf) * 1000.0

# Compute area-averaged values
P_v = P_gridVal.mean(dim=["latitude", "longitude"]).values
ET_v = ET_gridVal.mean(dim=["latitude", "longitude"]).values
Q_obs_v = Q_gridVal.mean(dim=["latitude", "longitude"]).values

# Ensure ET is positive
ET_v = np.where(ET_v < 0.0, -ET_v, ET_v) 

Q_sim_v = simulate_runoff(best_k, P_v, ET_v)
#print(Q_obs_v, Q_sim_v)
kge_v = kge(Q_obs_v, Q_sim_v)

print(f"KGE for validation: {kge_v[0]:.3f}")

print(f"r = {kge_v[1]:.3f}, alpha = {kge_v[2]:.3f}, beta = {kge_v[3]:.3f}")

#Ploting time series for 2002 data 
Q_gridVal_mean = Q_gridVal.mean(dim=["latitude", "longitude"])
P_v_mean = P_gridVal.mean(dim=["latitude", "longitude"])
ET_v_mean = ET_gridVal.mean(dim=["latitude", "longitude"])*-1

Q_df = Q_gridVal_mean.to_dataframe()
P_df = P_v_mean.to_dataframe()
ET_df = ET_v_mean.to_dataframe()

Q_df.reset_index(inplace=True)
P_df.reset_index(inplace=True)
ET_df.reset_index(inplace=True)

df = pd.DataFrame({'Time': Q_df['valid_time'], 'Runoff': Q_df['ro'], 'Rainfall': P_df['tp'],
'Evaporation': ET_df['e']})

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(df['Time'], df['Rainfall'], marker='o', linestyle='--', color='b', label='Rainfall')
plt.plot(df['Time'], df['Runoff'], marker='s', linestyle='-', color='r', label='Runoff')
plt.plot(df['Time'], df['Evaporation'], marker='', linestyle='-', color='g', label='Evaporation')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('mm')
plt.title('Hourly precipitation, total evaporation, and runoff - 2002')
plt.legend()

plt.savefig('ERA5_2002.png', dpi=300, bbox_inches='tight')

#Ploting time series for 2001 data
Q_mean = Q_grid.mean(dim=["latitude", "longitude"])
P_mean = P_grid.mean(dim=["latitude", "longitude"])
ET_mean = ET_grid.mean(dim=["latitude", "longitude"])*-1

Q_df_2001 = Q_mean.to_dataframe()
P_df_2001 = P_mean.to_dataframe()
ET_df_2001 = ET_mean.to_dataframe()

Q_df_2001.reset_index(inplace=True)
P_df_2001.reset_index(inplace=True)
ET_df_2001.reset_index(inplace=True)

df_2001 = pd.DataFrame({'Time': Q_df_2001['valid_time'], 'Runoff': Q_df_2001['ro'], 'Rainfall': P_df_2001['tp'],
'Evaporation': ET_df_2001['e']})

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(df_2001['Time'], df_2001['Rainfall'], marker='o', linestyle='--', color='b', label='Rainfall')
plt.plot(df_2001['Time'], df_2001['Runoff'], marker='s', linestyle='-', color='r', label='Runoff')
plt.plot(df_2001['Time'], df_2001['Evaporation'], marker='', linestyle='-', color='g', label='Evaporation')

# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('mm')
plt.title('Hourly precipitation, total evaporation, and runoff - 2001')
plt.legend()

plt.savefig('ERA5-2001.png', dpi=300, bbox_inches='tight')

#Data frame with different values of k for 2001 calibrating period 
df_Q = pd.DataFrame({'Time': Q_df_2001['valid_time'], 'Observed Runoff': Q_df_2001['ro'],
'Best K (0.509) Simulated Runoff': Q_sim, 'Simulated Runoff - K(0.15)': Qsim})

# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(df_Q['Time'], df_Q['Observed Runoff'], marker='s', linestyle='-', color='b', label='Observed Runoff')
plt.plot(df_2001['Time'], df_Q['Best K (0.509) Simulated Runoff'], marker='', linestyle='-', color='r', label='Best K (0.509) Simulated Runoff')
plt.plot(df_2001['Time'], df_Q['Simulated Runoff - K(0.15)'], marker='', linestyle='-', color='g', label='Simulated Runoff - K(0.15)')
# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('mm')
plt.title('Observed and Simulated runoff model - 2001')
plt.legend()
plt.savefig('ValidatedK.png', dpi=300, bbox_inches='tight')

#Sacatter plot for best k in calibration data 2001
plt.figure(figsize=(8, 8))
plt.scatter(df_Q['Observed Runoff'], df_Q['Best K (0.509) Simulated Runoff'], color='b', alpha= 0.5, label = 'data points')

# Add 1:1 line
max_val = max(max(df_Q['Observed Runoff']), max(df_Q['Best K (0.509) Simulated Runoff']))
plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')  # Red dashed line

# Labels and title
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Best K (0.509) Simulated Runoff (mm)')
plt.title('Simulated vs. Observed Runoff 2001')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('scatter_obs_sim_2001.png', dpi=300, bbox_inches='tight')

#Sacatter plot with initial k in calibration data 2001
plt.figure(figsize=(8, 8))
plt.scatter(df_Q['Observed Runoff'], df_Q['Simulated Runoff - K(0.15)'], color='b', alpha= 0.5, label = 'data points')

# Add 1:1 line
max_val = max(max(df_Q['Observed Runoff']), max(df_Q['Simulated Runoff - K(0.15)']))
plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')  # Red dashed line

# Labels and title
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Simulated Runoff - K(0.15) (mm)')
plt.title('Simulated vs. Observed Runoff 2001')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('scatter_osv_sim_2001_k.png', dpi=300, bbox_inches='tight')

#Data frame with best k for 2002 validation period 
df_Q_v = pd.DataFrame({'Time': Q_df['valid_time'], 'Observed Runoff 2002': Q_obs_v, 
'Best K (0.509) Simulated Runoff for validation': Q_sim_v})

#Time series with observed and simulated 2002 runoff values for validation 
# Create a figure and axis
plt.figure(figsize=(20, 6))

# Plot  Averages
plt.plot(df_Q_v['Time'], df_Q_v['Observed Runoff 2002'], marker='s', linestyle='-', color='b', label='Observed Runoff 2002')
plt.plot(df_Q_v['Time'], df_Q_v['Best K (0.509) Simulated Runoff for validation'], marker='', linestyle='-', color='r', 
label='Best K (0.509) Simulated Runoff for validation')
# Add labels, title, and legend
plt.xlabel('Time')
plt.ylabel('mm')
plt.title('Observed and Simulated runoff model for validation - 2002')
plt.legend()
plt.savefig('time_seriesQ-2002_bestK.png', dpi=300, bbox_inches='tight')

##Sacatter plot with best k in validation data 2002
plt.figure(figsize=(8, 8))
plt.scatter(df_Q_v['Observed Runoff 2002'], df_Q_v['Best K (0.509) Simulated Runoff for validation'], color='b', alpha= 0.5, label = 'data points')

# Add 1:1 line
max_val = max(max(df_Q_v['Observed Runoff 2002']), max(df_Q_v['Best K (0.509) Simulated Runoff for validation']))
plt.plot([0, max_val], [0, max_val], 'r--', label='1:1 Line')  # Red dashed line

# Labels and title
plt.xlabel('Observed Runoff (mm)')
plt.ylabel('Best K (0.509) Simulated Runoff for validation')
plt.title('Simulated vs. Observed Runoff for validation -2002')
plt.legend()
plt.grid(True)
#plt.show()
plt.savefig('scatter_osv_sim_2002.png', dpi=300, bbox_inches='tight')