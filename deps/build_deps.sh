#!/bin/bash
cd "$(dirname "$0")" || exit 1
shopt -s nocasematch

COMPILE_THREADS=${COMPILE_THREADS:-6}

SUNDIALS_VER="5.7.0"

mkdir -p downloads
cd downloads || exit 1
wget -nc "https://github.com/LLNL/sundials/releases/download/v$SUNDIALS_VER/sundials-$SUNDIALS_VER.tar.gz"

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

tar xvf "../downloads/sundials-$SUNDIALS_VER.tar.gz"

cd "sundials-$SUNDIALS_VER" || exit 1
mkdir -p build
cd build || exit 1

if [[ -n "$DEBUG" ]]; then
    cmake .. \
        -DCMAKE_BUILD_TYPE=Debug \
        -DENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX="$PWD/../../"
else
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DENABLE_OPENMP=ON -DCMAKE_INSTALL_PREFIX="$PWD/../../"
fi

make -j "$COMPILE_THREADS"
make install
