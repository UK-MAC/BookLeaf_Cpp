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

## Building

BookLeaf uses CMake, and tries to be idiomatic in doing so. The typical CMake
process looks something like the following (inside the top-level BookLeaf
directory):

```
mkdir build
cd build
cmake \
    -DCMAKE_INSTALL_PREFIX=$HOME \
    -DCMAKE_BUILD_TYPE="Release" \
    -DENABLE_TYPHON=ON \
    -DENABLE_PARMETIS=ON \
    -DENABLE_SILO=ON \
    -DENABLE_CALIPER=ON \
    ..
make
make test
```

`make test` will run both unit tests for individual kernels, and (TODO
validation tests for the entire application), but requires building with zlib
(which is the default).
