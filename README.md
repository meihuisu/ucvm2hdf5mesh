# ucvm2hdf5mesh

hdf5 mesh generator using UCVM api to fill in the material properties

    require mpi

Instruction for running on ORNL's Frontier

module list,

  # UCVM/sfcvm,  sw4
  module load cray-python
  module unload PrgEnv-cray 
  module load PrgEnv-gnu gcc
  module load Core/24.07
  module load libtool/2.4.6
  module load openblas/0.3.26

  #https://docs.olcf.ornl.gov/software/python/parallel_h5py.html
  module load cray-hdf5-parallel

Using conda env to setup running environment

needs to load python version of hdf5  
  hdf5py
needs to load python --..



