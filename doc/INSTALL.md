# Manual installation

Installation using Spack is recommended, see [README.md](../README.md). This
file details how to build and install BookLeaf directly using CMake.

## Dependencies

The only hard dependency is [yaml-cpp](https://github.com/jbeder/yaml-cpp),
which is used to read input files.

If you want to run BookLeaf using MPI (the typical case), then you will need to
enable Typhon support by first installing that library and then passing the
`ENABLE_TYPHON` switch to CMake (as shown below).

ParMETIS and Silo are optional dependencies which accelerate mesh partitioning
and provide visualisation files respectively.

LLNL Caliper is supported for collecting performance measurements, build with
`ENABLE_CALIPER` to enable this.

zlib is used for reading and writing compressed dumps, and is required for
running tests. CMake will try and locate zlib by default, but will not stop the
build if it cannot find it.

* yaml-cpp (https://github.com/jbeder/yaml-cpp)
* (OPTIONAL) Typhon + MPI (https://bitbucket.org/warwickhpsc/typhon)
* (OPTIONAL) Silo + HDF5 (https://wci.llnl.gov/simulation/computer-codes/silo)
* (OPTIONAL) METIS + ParMETIS (http://glaros.dtc.umn.edu/gkhome/metis/metis/overview)
* (OPTIONAL) Caliper (https://github.com/LLNL/Caliper)
* (OPTIONAL) zlib (https://www.zlib.net)

## CUDA Dependencies

The CUDA version of BookLeaf additionally requires CUDA itself, and [NVIDIA
CUB](https://nvlabs.github.io/cub/).

## Building

BookLeaf uses CMake, and tries to be idiomatic in doing so.  We recommend using
the `nvcc_wrapper` script available
[here](https://github.com/kokkos/nvcc_wrapper) to build the CUDA variant as this
dramatically simplifies managing the compiler options. This build process using
`nvcc_wrapper` looks something like the following (where `sm_70` is replaced
with the CUDA architecture of the NVIDIA GPU you wish to run on):

```
mkdir build
cd build
export CXX=nvcc_wrapper
cmake \
    -DCMAKE_INSTALL_PREFIX=$HOME \
    -DCMAKE_BUILD_TYPE="Release" \
    -DCMAKE_CXX_FLAGS="-ccbin <path to host compiler> -arch sm_70" \
    -DYamlCpp_ROOT_DIR=<path to install> \
    -DCUB_ROOT_DIR=<path to install> \
    -DENABLE_TYPHON=ON \
    -DTyphon_ROOT_DIR=<path to install> \
    -DENABLE_PARMETIS=ON \
    -DMETIS_ROOT_DIR=<path to install> \
    -DParMETIS_ROOT_DIR=<path to install> \
    -DENABLE_SILO=ON \
    -DHDF5_ROOT=<path to install> \
    -DSilo_ROOT_DIR=<path to install> \
    -DENABLE_CALIPER=ON \
    -DCaliper_ROOT_DIR=<path to install> \
    ..
make
```

BookLeaf's unit tests currently do not work with the CUDA version.
