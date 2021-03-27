#!/bin/bash
cd "$(dirname "$0")" || exit 1
shopt -s nocasematch

SUNDIALS_LIB_LIST="cvode nvecserial"

COMPILE_THREADS=${COMPILE_THREADS:-6}
CFLAGS_RELEASE="-Ofast -march=native -fopenmp -flto=$COMPILE_THREADS"
CFLAGS_DEBUG="-Og -g -fopenmp"
SRC="main.c"
OUT="bin/main"

LIBS=" -lm"
for lib in $SUNDIALS_LIB_LIST; do
    LIBS+=" -lsundials_$lib"
done

if [[ $1 == "DEBUG" || -n "$DEBUG" ]]; then
    echo "Debug build"
    DEBUG=1

    CFLAGS="$CFLAGS_RELEASE"
    DEPS="deps/debug/"
else
    echo "Normal build"

    CFLAGS="$CFLAGS_DEBUG"
    DEPS="deps/release/"
fi

if [[ ! -d "$DEPS" ]]; then
    echo "Dependencies not found, rebuilding"
    DEBUG="$DEBUG" deps/build_deps.sh
fi

gcc $CFLAGS \
    -I "$DEPS/include" $SRC -o "$OUT" -L "$DEPS/lib" \
    $LIBS -Wl,-rpath="\$ORIGIN/../$DEPS/lib"
