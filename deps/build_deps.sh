#!/bin/bash
cd "$(dirname "$0")" || exit 1
shopt -s nocasematch

COMPILE_THREADS=${COMPILE_THREADS:-6}

OPEN_BLAS_VER="0.3.13"
SUNDIALS_VER="5.7.0"
ARKODE_VER="4.7.0"

ARKODE_RELEASE_FLAGS="-Ofast -march=native -flto=$COMPILE_THREADS"

mkdir -p downloads
cd downloads || exit 1

wget -nc "https://github.com/xianyi/OpenBLAS/releases/download/v$OPEN_BLAS_VER/OpenBLAS-$OPEN_BLAS_VER.tar.gz"
wget -nc "https://github.com/LLNL/sundials/releases/download/v$SUNDIALS_VER/arkode-$ARKODE_VER.tar.gz"

cd .. || exit 1

if [[ $1 == "DEBUG" || -n "$DEBUG" ]]; then
    DEBUG=1
    mkdir -p debug
    cd debug || exit 1
else
    mkdir -p release
    cd release || exit 1
fi

mkdir -p lib
ln -s lib lib64
mkdir -p downloads

tar xvf "../downloads/OpenBLAS-$OPEN_BLAS_VER.tar.gz"
tar xvf "../downloads/arkode-$ARKODE_VER.tar.gz"

cd "OpenBLAS-$OPEN_BLAS_VER" || exit 1

if [[ -n "$DEBUG" ]]; then
    make -j "$COMPILE_THREADS" \
        DEBUG=1 \
        BUILD_RELAPACK=1 USE_OPENMP=1
else
    make -j "$COMPILE_THREADS" BUILD_RELAPACK=1 USE_OPENMP=1
fi

make PREFIX="../" install

cd "../arkode-$ARKODE_VER" || exit 1
mkdir -p build
cd build || exit 1

if [[ -n "$DEBUG" ]]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DLAPACK_LIBRARIES="$PWD/../../lib/libopenblas.so" \
        -DENABLE_LAPACK=ON -DENABLE_OPENMP=ON \
        -DCMAKE_INSTALL_PREFIX="$PWD/../../"
else
    cmake .. \
        -DCMAKE_C_FLAGS="$ARKODE_RELEASE_FLAGS" \
        -DCMAKE_Fortran_FLAGS="$ARKODE_RELEASE_FLAGS" \
        -DLAPACK_LIBRARIES="$PWD/../../lib/libopenblas.so" \
        -DENABLE_LAPACK=ON -DENABLE_OPENMP=ON \
        -DCMAKE_INSTALL_PREFIX="$PWD/../../"
fi

make -j "$COMPILE_THREADS"
make install
