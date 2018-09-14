#!/bin/bash

set -e

if [ -f $HDF5_DIR/lib/libhdf5.so ]; then
    echo "Using cached build"
else
    pushd /tmp
    wget https://www.hdfgroup.org/ftp/HDF5/releases/hdf5-1.10/hdf5-$HDF5_VERSION/src/hdf5-$HDF5_VERSION.tar.gz
    tar -xzvf hdf5-$HDF5_VERSION.tar.gz
    pushd hdf5-$HDF5_VERSION
    chmod u+x autogen.sh
    ./configure --prefix $HDF5_DIR
    make -j 2
    make install
    popd
    popd
fi
