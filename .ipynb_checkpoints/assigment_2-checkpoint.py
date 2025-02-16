#Import the necessary libraries to read the climate data
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr


#Open the dataset by introducing the correct path to the folder
dset = xr.open_dataset(r'C:\Users\mbarc\geo_env\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_195001-201412.nc')


#Covert the dataset to a numpy array and explore it 
Clim = np.array(dset.variables)

Clim


ClimT = dset['tas']


ClimT


dset['tas'].dtype


#Open dataset to calculate the mean temperature between 1850-1900 and produce the map of the mean temperature at preindustrial levels 


dset = xr.open_dataset(r'C:\Users\mbarc\geo_env\Climate_Model_Data\tas_Amon_GFDL-ESM4_historical_r1i1p1f1_gr1_185001-194912.nc')


dset['tas']


mean_1850_1900 = np.mean(dset['tas'].sel(time=slice('18500101','19001231')), axis=0)


mean_1850_1900 = np.array(mean_1850_1900)

plt.imshow(mean_1850_1900, cmap='coolwarm', origin='lower')
cbar = plt.colorbar().set_label('Tempeture (K)')
plt.title("Temperature Preindustrial 1850-1900)")
plt.savefig('preindustrial3.png', dpi=300, bbox_inches='tight')
plt.show()


# Open the dataset for the different socioeconomical scenarios and compute the mean temperature between 2071 and 2100 


dset_119 = xr.open_dataset(r'C:\Users\mbarc\geo_env\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp119_r1i1p1f1_gr1_201501-210012.nc')


mean_2071_2100_ssp119 = np.mean(dset_119['tas'].sel(time=slice('20710101','21001231')), axis=0)


mean_2071_2100_ssp119 = np.array(mean_2071_2100_ssp119)
mean_2071_2100_ssp119


# Compute the difference between the preindustrial temperature level and the mean temperature at ssp119 and produce the color map for it 


delta_temp = mean_2071_2100_ssp119 - mean_1850_1900
plt.imshow(delta_temp, cmap='RdBu_r', origin='lower')
plt.colorbar(label="Temperature Change (K)")
plt.title("Temperature Difference (SSP21-1.9 - Preindustrial)")
plt.savefig('T_DiffSPP119_preindustrial3.png', dpi=300, bbox_inches='tight')
plt.show()


# Open the dataset for the different socioeconomical scenarios and compute the mean temperature between 2071 and 2100 
dset_245 = xr.open_dataset(r'C:\Users\mbarc\geo_env\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp245_r1i1p1f1_gr1_201501-210012.nc')

mean_2071_2100_ssp245 = np.mean(dset_245['tas'].sel(time=slice('20710101','21001231')), axis=0)

mean_2071_2100_ssp245


# Compute the difference between the preindustrial temperature level and the mean temperature at ssp245 and produce the color map for it 


delta_temp = mean_2071_2100_ssp245 - mean_1850_1900
plt.imshow(delta_temp, cmap='RdBu_r', origin='lower')
plt.colorbar(label="Temperature Change (K)")
plt.title("Temperature Difference (SSP2-4.5 - Preindustrial)")
plt.savefig('T_DiffSPP245_preindustrial3.png', dpi=300, bbox_inches='tight')
plt.show()


# Open the dataset for the different socioeconomical scenarios and compute the mean temperature between 2071 and 2100 

dset_585 = xr.open_dataset(r'C:\Users\mbarc\geo_env\Climate_Model_Data\tas_Amon_GFDL-ESM4_ssp585_r1i1p1f1_gr1_201501-210012.nc')


# Compute the difference between the preindustrial temperature level and the mean temperature at ssp585 and produce the color map for it 


mean_2071_2100_ssp585 = np.mean(dset_585['tas'].sel(time=slice('20710101','21001231')), axis=0)



delta_temp = mean_2071_2100_ssp585 - mean_1850_1900
plt.imshow(delta_temp, cmap='RdBu_r', origin='lower')
plt.colorbar(label="Temperature Change (K)")
plt.title("Temperature Difference (SSP5-8.5 - Preindustrial)")
plt.savefig('T_DiffSPP585_preindustrial3.png', dpi=300, bbox_inches='tight')
plt.show()







