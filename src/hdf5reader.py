#!/usr/bin/env python
#
#  camilo's hdf5 reader
#

import h5py
import pyproj
import os
import numpy as np
from cvm_ucvm import UCVM, Point
from cvm_hdf5 import coord_system_array, local_coords_getter


#  rank: 2584  grid: grid_0  ii> 2584  iii> 0  with  21  latlon
#(-120.8320, 37.2769, 0.0000)
def test_target_point(h5_file,shapes):
    query_lon=-120.8320
    query_lat=37.2769
    test_point(h5_file, shapes, query_lon, query_lat)

#(-122.04519,37.4161)
def test_s3240_point(h5_file,shapes):
    query_lon=-122.04519
    query_lat=37.4161
    test_point(h5_file, shapes, query_lon, query_lat)

#  rank: 2584  grid: grid_0  ii> 2584  iii> 0  with  21  latlon
#(-120.8320, 37.2769, 0.0000)
def test_point(h5_file, shapes, query_lon, query_lat):
    top= h5_file['/']
    lon_anchor, lat_anchor, azimuth = top.attrs['Origin longitude, latitude, azimuth']
    print("lon_anchor>>>",lon_anchor)
    print("lat_anchor>>>",lat_anchor)
    print("azimuth>>>",azimuth)
    print("query_lon>>>",query_lon)
    print("query_lat>>>",query_lat)

## try grid_0
    grid="grid_0"
    length, width = 280000, 140000
    nx,ny,nz=shapes[0]
    dx, dy = length // (nx - 1), width // (ny - 1)

    x_coord, y_coord=local_coords_getter(query_lon, query_lat, lon_anchor, lat_anchor, azimuth)
    coord_system= coord_system_array(int(length), int(width), dx, dy, azimuth, lon_anchor, lat_anchor)
    target_x_idx = np.argmin(np.abs(coord_system.x_axis.values - x_coord))
    target_y_idx = np.argmin(np.abs(coord_system.y_axis.values - y_coord))

    print("target_x_idx>>>",target_x_idx)
    print("target_y_idx>>>",target_y_idx)

    matprop_vs=h5_file['Material_model'][grid]['Cs'][target_x_idx, target_y_idx, 0]
    matprop_vp=h5_file['Material_model'][grid]['Cp'][target_x_idx, target_y_idx, 0]
    matprop_density=h5_file['Material_model'][grid]['Rho'][target_x_idx, target_y_idx, 0]
    print("matprop_vs>>>", matprop_vs)
    print("matprop_vp>>>", matprop_vp)
    print("matprop_density>>>", matprop_density)


def read_h5file(filename) :

    print("..open HDF5 file..")
    h5_file = h5py.File(f'{filename}.h5', 'r')
    shapes=[]

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
        nx,ny,nz=tmpCp.shape
        shapes.append([nx,ny,nz])
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

    return h5_file, shapes


if __name__ == '__main__':

    filename = 'cs248_mpi_sfile_mei'

    handle, shapes=read_h5file(filename)
    
    if 'UCVM_INSTALL_PATH' in os.environ:
        installdir = os.environ.get('UCVM_INSTALL_PATH')
    ucvm_model = UCVM(install_dir=installdir)

    print("\n...LOOKING at s3240")
    test_s3240_point(handle,shapes)
    print("\n...LOOKING at target")
    test_target_point(handle,shapes)

    handle.close()

    print('\nDone..')
    
