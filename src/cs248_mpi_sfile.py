#!/usr/bin/env python
#
#  hdf5 mesh generator using UCVM api to fill in the material properties
#  require mpi
#

import h5py
import numpy as np
import pyproj
import xarray as xr
from cvm_ucvm import UCVM, Point
import os
import datetime

from mpi4py import MPI
import h5py

mycomm = MPI.COMM_WORLD    # Use the world communicator
myrank = mycomm.Get_rank() # The process ID (integer 0-41 for a 42-process job)
mysize = mycomm.Get_size() # Total amount of ranks


def create_h5file(azimuth, lon_anchor, lat_anchor, filename, blocks_info, n_comp=5):
    print("Creating h5 header..")
    h5_file = h5py.File(f'{filename}.h5', 'w')
    h5_file.create_group('Material_model')
    h5_file.create_group('Z_interfaces')

    n_grids = blocks_info.size

    h5_file['/'].attrs['Attenuation'] = 1
    h5_file['/'].attrs['Coarsest horizontal grid spacing'] = blocks_info['hres'].max()
    h5_file['/'].attrs['Min, max depth'] = blocks_info['z_top'].min(), blocks_info['z_bot'].max()
    h5_file['/'].attrs['Origin longitude, latitude, azimuth'] = lon_anchor, lat_anchor, azimuth
    h5_file['/'].attrs['ngrids'] = n_grids

    for i in range(n_grids):
        h5_file['Material_model'].create_group(f'grid_{i}')
        h5_file['Material_model'][f'grid_{i}'].attrs['Horizontal grid size'] = blocks_info['hres'][i]
        h5_file['Material_model'][f'grid_{i}'].attrs['Number of components'] = n_comp

        nx, ny, nz = blocks_info[['nx', 'ny', 'nz']][i]

        h5_file['Material_model'][f'grid_{i}'].create_dataset('Cp', data=np.zeros((nx, ny, nz)))
        h5_file['Material_model'][f'grid_{i}'].create_dataset('Cs', data=np.zeros((nx, ny, nz)))
        h5_file['Material_model'][f'grid_{i}'].create_dataset('Qp', data=np.zeros((nx, ny, nz)))
        h5_file['Material_model'][f'grid_{i}'].create_dataset('Qs', data=np.zeros((nx, ny, nz)))
        h5_file['Material_model'][f'grid_{i}'].create_dataset('Rho', data=np.zeros((nx, ny, nz)))

    for i in range(n_grids):
        nx, ny = blocks_info[['nx', 'ny']][i]
        if i == 0:
            h5_file['Z_interfaces'].create_dataset(f'z_values_{i}', data=np.zeros((nx, ny)))
            h5_file['Z_interfaces'][f'z_values_{i}'][:, :] = blocks_info['z_top'][i]

        h5_file['Z_interfaces'].create_dataset(f'z_values_{i + 1}', data=np.zeros((nx, ny)))
        h5_file['Z_interfaces'][f'z_values_{i + 1}'][:, :] = blocks_info['z_bot'][i]

    h5_file.close()


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
        coordinates_array.loc[:, :, 'rotated_strike'] = (coordinates_array.loc[:, :, 'local_strike'] * np.cos(rad_az) -
                                                         coordinates_array.loc[:, :, 'local_dip'] * np.sin(rad_az))
        coordinates_array.loc[:, :, 'rotated_dip'] = (coordinates_array.loc[:, :, 'local_strike'] * np.sin(rad_az) +
                                                      coordinates_array.loc[:, :, 'local_dip'] * np.cos(rad_az))

        dist = np.sqrt(coordinates_array.loc[:, :, 'rotated_strike'].values ** 2 + coordinates_array.loc[:, :,
                                                                                   'rotated_dip'].values ** 2)
        az12 = np.rad2deg(np.arctan2(coordinates_array.loc[:, :, 'rotated_dip'].values,
                                     coordinates_array.loc[:, :, 'rotated_strike'].values + 0.0000001))

        # GRS80 is the ellipsoid used by NAD83
        tr = pyproj.Geod(ellps='GRS80')
        endlon, endlat, backaz = tr.fwd(np.full_like(coordinates_array.loc[:, :, 'rotated_strike'].values, lon_anchor),
                                        np.full_like(coordinates_array.loc[:, :, 'rotated_dip'].values, lat_anchor),
                                        az12, dist)
        coordinates_array.loc[:, :, 'lon'] = endlon
        coordinates_array.loc[:, :, 'lat'] = endlat

        return coordinates_array


def block_info(n_blocks, length, width):
    header = [('grid', 'i4'), ('hres', 'i4'), ('vres', 'i4'), ('nx', 'i4'), ('ny', 'i4'), ('nz', 'i4'), ('z_top', 'f8'), ('z_bot', 'f8')]

    block_array = np.zeros(n_blocks, dtype=header)
    block_array['grid'] = np.arange(n_blocks)
    block_array['hres'] = [100, 200, 500, 1000]
    block_array['vres'] = [25, 50, 125, 250]
    block_array['z_top'] = [0, 500, 3500, 10000]
    block_array['z_bot'] = [500, 3500, 10000, 30000]

    for i in range(n_blocks):
        block_array['nx'][i] = length // block_array['hres'][i] + 1
        block_array['ny'][i] = width // block_array['hres'][i] + 1
        block_array['nz'][i] = np.abs(block_array['z_top'][i] - block_array['z_bot'][i]) // block_array['vres'][i] + 1

    return block_array


def populate_hdf5(_grid,_rank,_comm, filename, blocks_info, length, width, azimuth, lon_anchor, lat_anchor, ucvm):
    h5_file = h5py.File(f'{filename}.h5', 'r+', driver='mpio', comm=MPI.COMM_WORLD)

    print(">>> populate  for rank",_rank)
    if blocks_info.size != len(h5_file['Material_model']):
        raise ValueError('There is an inconsistency between number of blocks of the array and hdf5 file.')

    i=_grid ## just do the first grid

    grid="grid_%d"%(i)
    #for i, grid in enumerate(h5_file['Material_model']):

    print(grid + "for rank>", _rank )
    nx, ny, nz, z_top ,z_bot = blocks_info[['nx', 'ny', 'nz', 'z_top', 'z_bot']][i]
    dx, dy = length // (nx - 1), width // (ny - 1)
    depth_vector = np.linspace(z_top, z_bot, nz)
    coords_layer = coord_system_array(length, width, dx, dy, azimuth, lon_anchor, lat_anchor)

##    print("ii range is ",range(coords_layer.shape[0]))
##    print("iii range is ",range(coords_layer.shape[1]))
    ii=_rank
    max_ii=coords_layer.shape[0]
    if (ii < max_ii) :
      for iii in range(coords_layer.shape[1]):
        lon_query = coords_layer.loc[:, :, 'lon'][ii, iii]
        lat_query = coords_layer.loc[:, :, 'lat'][ii, iii]

        ucvmpoints = []
        for depth in depth_vector:
            ucvmpoints.append(Point(lon_query, lat_query, depth))

##        print(" rank:", _rank, " grid:", grid, " ii>",ii," iii>", iii," with ",len(ucvmpoints)," latlon")

        data = ucvm.query(ucvmpoints, "cs248")
    
        for j, matprop in enumerate(data):

            if(matprop.vs < 400):
              h5_file['Material_model'][grid]['Cs'][ii, iii, j] = 400
              h5_file['Material_model'][grid]['Qs'][ii, iii, j] = 400 / 20 # QS=VS/10???
              h5_file['Material_model'][grid]['Qp'][ii, iii, j] = 2 * 400 / 20 # QP = 2 QS
            else:
              h5_file['Material_model'][grid]['Cs'][ii, iii, j] = matprop.vs # VS
              h5_file['Material_model'][grid]['Qs'][ii, iii, j] = matprop.vs / 20 # QS=VS/10???
              h5_file['Material_model'][grid]['Qp'][ii, iii, j] = 2 * matprop.vs / 20 # QP = 2 QS
              
            if(matprop.vp < 800):
              h5_file['Material_model'][grid]['Cp'][ii, iii, j] = 800
            else:
              h5_file['Material_model'][grid]['Cp'][ii, iii, j] = matprop.vp # VP

            if(matprop.density < 1800) :
              h5_file['Material_model'][grid]['Rho'][ii, iii, j] = 1800
            else:
              h5_file['Material_model'][grid]['Rho'][ii, iii, j] = matprop.density # DENSITY


    h5_file.close()


if __name__ == '__main__':
    if (myrank==0) :
      ct = datetime.datetime.now()
      print("maker start time >",ct)

    if 'UCVM_INSTALL_PATH' in os.environ:
        installdir = os.environ.get('UCVM_INSTALL_PATH')
    ucvm_model = UCVM(install_dir=installdir)

    # VOLUME LENGTH AND WIDTH AND NUMBER OF GRIDS IN THE HDF5 FILE
    length, width, n_blocks = 280000, 140000, 4
    # AZIMUTH OF ROTATION AND ANCHOR POINT COORDINATES
    azimuth, lon_anchor, lat_anchor = 143.638, -122.559194, 39.164488

    # HDF5 FILE NAME
    filename = 'cs248_mpi_sfile'

    # FUNCTION WITH THE DATA OF DISCRETIZE THE MESH
    block_array = block_info(n_blocks, length, width)

     # CREATE AN EMPTY HDF5 FILE
    if (myrank==0) :
      create_h5file(azimuth, lon_anchor, lat_anchor, filename, block_array, n_comp=5)

    mycomm.Barrier()
    if (myrank==0) :
      ctp = datetime.datetime.now()
      print("maker start populate time >",ctp)


    for mygrid in range(n_blocks) :
    # POPULATE THE HDF5 WITH THE GIVEN MATERIAL PROPERTIES
      populate_hdf5(mygrid,myrank,mycomm, filename, block_array, length, width, azimuth, lon_anchor, lat_anchor, ucvm_model)
      mycomm.Barrier()

    mycomm.Barrier()

    if(myrank == 0):
      ctpp = datetime.datetime.now()
      print("maker end time >",ctpp)

