#!/usr/bin/env python
##
#  @file cvm_hdf5.py
#  @brief Common definitions and functions for hdf5-cvm
#  hdf5 mesh utilities
#
import sys
import os
import pyproj
import math
import struct
import xarray as xr

#  Numpy is required.
try:
    import numpy as np
except:
    print("ERROR: NumPy must be installed on your system in order to generate these plots.")
    exit(1)

def coord_system_array(length, width, dx, dy, azimuth, lon_anchor, lat_anchor):
    coords_strike_axis = np.linspace(0, length, length // dx + 1)
    coords_dip_axis = np.linspace(0, width, width // dy + 1)

    coordinates_array = xr.DataArray(np.zeros((coords_strike_axis.size, coords_dip_axis.size, 6)),
                     dims=('x_axis', 'y_axis', 'parameter'),
                     coords={'x_axis': coords_strike_axis,
                             'y_axis': coords_dip_axis,
                              'parameter': ['local_strike', 'local_dip', 'rotated_strike',
                                            'rotated_dip', 'lon', 'lat']})

    coordinates_array.loc[:, :, 'local_strike'] = np.array(
        [coords_strike_axis for _ in range(coords_dip_axis.size)]).T
    coordinates_array.loc[:, :, 'local_dip'] = np.array([coords_dip_axis for _ in range(coords_strike_axis.size)])

    rad_az = np.deg2rad(azimuth)
    coordinates_array.loc[:, :, 'rotated_strike'] = (coordinates_array.loc[:, :, 'local_strike'] * np.cos(rad_az) 
          - coordinates_array.loc[:, :, 'local_dip'] * np.sin(rad_az))
    coordinates_array.loc[:, :, 'rotated_dip'] = (coordinates_array.loc[:, :, 'local_strike'] * np.sin(rad_az) 
          + coordinates_array.loc[:, :, 'local_dip'] * np.cos(rad_az))

    dist = np.sqrt(coordinates_array.loc[:, :, 'rotated_strike'].values ** 2 
                   + coordinates_array.loc[:, :,
                                           'rotated_dip'].values ** 2)
    az12 = np.rad2deg(np.arctan2(coordinates_array.loc[:, :, 'rotated_dip'].values,
                   coordinates_array.loc[:, :, 'rotated_strike'].values + 0.0000001))

    # GRS80 is the ellipsoid used by NAD83
    tr = pyproj.Geod(ellps='GRS80')
    endlon, endlat, backaz = \
                    tr.fwd(np.full_like(coordinates_array.loc[:, :, 'rotated_strike'].values, lon_anchor),
                    np.full_like(coordinates_array.loc[:, :, 'rotated_dip'].values, lat_anchor),
                    az12, dist)
    coordinates_array.loc[:, :, 'lon'] = endlon
    coordinates_array.loc[:, :, 'lat'] = endlat

    return coordinates_array

def local_coords_getter(query_lon, query_lat, lon_anchor, lat_anchor, azimuth):
   tr = pyproj.Geod(ellps='GRS80')
   az12, _, dist = tr.inv(np.full_like(query_lon, lon_anchor), np.full_like(query_lat, lat_anchor),
                                   query_lon, query_lat)
   x_rotated = dist * np.sin(np.deg2rad(az12))
   y_rotated = dist * np.cos(np.deg2rad(az12))

   rad_az = np.deg2rad(azimuth)
   x_local = (x_rotated * np.sin(rad_az) + y_rotated * np.cos(rad_az))
   y_local = (x_rotated * np.cos(rad_az) - y_rotated * np.sin(rad_az))

   return x_local, y_local


