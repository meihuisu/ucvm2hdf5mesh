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


def read_h5file(filename) :

    print("Open file header..")
    h5_file = h5py.File(f'{filename}.h5', 'r')

    print(h5_file.keys())

    for name in h5_file:
      print(name)

    mm = h5_file['Material_model']
    zi = h5_file['Z_interfaces']

    return h5_file
}


if __name__ == '__main__':

    filename = 'cs248_mpi_sfile'

    handle=read_h5file(filename)
    
    if 'UCVM_INSTALL_PATH' in os.environ:
        installdir = os.environ.get('UCVM_INSTALL_PATH')
    ucvm_model = UCVM(install_dir=installdir)

    handle.close()
    
