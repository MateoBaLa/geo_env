#Import necesary pakages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pdb
import xarray as xr

#Open dataset == "C:\Users\mbarc\geo_env\N21E039.SRTMGL1_NC.nc"
dset = xr.open_dataset(r'C:\Users\mbarc\geo_env\N21E039.SRTMGL1_NC.nc')
#Stop code and allow python script in the console
#pdb.set_trace()

#NumPy array to understand data extend, values, and type
DEM = np.array(dset.variables['SRTMGL1_DEM']) 
 
#dset.close()

#DEM Visualization parameters
plt.imshow(DEM)
cbar = plt.colorbar().set_label('Elevation (m asl)')
plt.show()
 
plt.savefig('assignment1.png', dpi=300)

git add assignment1.py
