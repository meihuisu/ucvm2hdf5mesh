#!/usr/bin/env python
#
#  camilo's hdf5 reader
#

import h5py
import numpy as np
import pyproj
import xarray as xr
from cvm_ucvm import UCVM, Point
import os
import datetime
import pdb


def local_coords_getter(query_lon, query_lat, lon_anchor, lat_anchor, azimuth):
    tr = pyproj.Geod(ellps='GRS80')
    az12, _, dist = tr.inv(np.full_like(query_lon, lon_anchor), np.full_like(query_lat, lat_anchor),
                                    query_lon, query_lat)
    x_rotated = dist * np.sin(np.deg2rad(az12))
    y_rotated = dist * np.cos(np.deg2rad(az12))

    rad_az = np.deg2rad(azimuth)
    x_local = (x_rotated * np.cos(rad_az) - y_rotated * np.sin(rad_az))
    y_local = (x_rotated * np.sin(rad_az) + y_rotated * np.cos(rad_az))

    return x_local, y_local

#  rank: 2584  grid: grid_0  ii> 2584  iii> 0  with  21  latlon
#(-120.8320, 37.2769, 0.0000)
def test_target_point(h5_file):
    query_lon=-120.8320
    query_lat=37.2769
    test_point(h5_file, query_lon, query_lat)

#(-122.04519,37.4161)
def test_s3240_point(h5_file):
    query_lon=-122.04519
    query_lat=37.4161
    test_point(h5_file, query_lon, query_lat)

//  rank: 2584  grid: grid_0  ii> 2584  iii> 0  with  21  latlon
//(-120.8320, 37.2769, 0.0000)
def test_point(h5_file, query_lon, query_lat):
    top= h5_file['/']
    lon_anchor, lat_anchor, azimuth = top.attrs['Origin longitude, latitude, azimuth']
    print("lon_anchor>>>",lon_anchor)
    print("lat_anchor>>>",lat_anchor)
    print("azimuth>>>",azimuth)
    print("query_lon>>>",query_lon)
    print("query_lat>>>",query_lat)
    x_local, y_local=local_coords_getter(query_lon, query_lat, lon_anchor, lat_anchor, azimuth)
    print("x_local>>>",x_local)
    print("y_local>>>",y_local)
    target_x=round(x_local)
    target_y=round(y_local)
    print("target_x>>>",target_x)
    print("target_y>>>",target_y)
## try grid_0
    grid="grid_0"
    vs=h5_file['Material_model'][grid]['Cs']
    matprops_vs=vs[target_x, target_y, 0]
    matprop_vs=h5_file['Material_model'][grid]['Cs'][target_x, target_y, 0]
    matprop_vp=h5_file['Material_model'][grid]['Cp'][target_x, target_y, 0]
    matprop_density=h5_file['Material_model'][grid]['Rho'][target_x, target_y, 0]
    print("matprop_vs>>>", matprop_vs)
    print("matprop_vp>>>", matprop_vp)
    print("matprop_density>>>", matprop_density)


def read_h5file(filename) :

    print("..open HDF5 file..")
    h5_file = h5py.File(f'{filename}.h5', 'r')

    print(h5_file.keys())

    print("group name:")
    for name in h5_file:
      print("   ",name)

    print("top attributes:")
    top= h5_file['/']
    print("  Attenuation:", top.attrs['Attenuation'])
    print("  Coarsest horizontal grid spacing:", top.attrs['Coarsest horizontal grid spacing'])
    print("  Min, max depth:", top.attrs['Min, max depth'])
    print("  Origin longitude, latitude, azimuth':", top.attrs['Origin longitude, latitude, azimuth'])
    print("  ngrids:", top.attrs['ngrids'])
    ngrids=top.attrs['ngrids']

    print("Material_model attributes:")
    mm = h5_file['Material_model']
    for i in range(ngrids):
        idx='grid_'+str(i)
        print("  for ",idx,":")
        tmp=mm[idx]
        print("    Horizontal grid size:", tmp.attrs['Horizontal grid size'])
        print("    Number of components:", tmp.attrs['Number of components'])

        tmpCp=tmp['Cp']
        if i == 3 :
          dslice=tmpCp[:1:1]
          print(dslice)
        tmpCs=tmp['Cs']
        tmpQp=tmp['Qp']
        tmpQs=tmp['Qs']
        tmpRho=tmp['Rho']

    print("Z_interface  attributes:")
    zi = h5_file['Z_interfaces']
    for i in range(ngrids):
        idx='z_values_'+str(i)
        idxx='z_values_'+str(i+1)
        print("  for ",idx,":")
        if i == 0:
           tmp0=zi[idx]
           z_top=tmp0[:,:]
           dslice=z_top[::1]
#           print(dslice)
        tmp1=zi[idxx]
        z_bot=tmp1[:,:]
        dslice=z_bot[::1]
#        print(dslice)

    return h5_file


if __name__ == '__main__':

    filename = 'cs248_mpi_sfile'

    handle=read_h5file(filename)
    
    if 'UCVM_INSTALL_PATH' in os.environ:
        installdir = os.environ.get('UCVM_INSTALL_PATH')
    ucvm_model = UCVM(install_dir=installdir)

    test_s3240_point(handle)

    handle.close()

    print('Done..')
    
