#!/bin/bash

set -euo pipefail

CMAKE_VERSION=3.26.1
LIBOMP_VERSION=15.0.7
FFTW_VERSION=3.3.10
CLINFO_VERSION=3.0.23.01.25
CCACHE_VERSION=4.8

mkdir -p toolchain
cd toolchain

PREFIX="$(pwd)"
NPROC=$(sysctl -n hw.physicalcpu)

export PATH="${PREFIX}/bin:${PATH+:$PATH}"
export LD_LIBRARY_PATH="${PREFIX}/lib${LD_LIBRARY_PATH+:$LD_LIBRARY_PATH}"

if [ -f ${PREFIX}/bin/cmake ] && ${PREFIX}/bin/cmake --version | grep -F ${CMAKE_VERSION} >/dev/null; then
    echo "CMake ${CMAKE_VERSION} already installed"
else
    echo "Installing CMake ${CMAKE_VERSION}"
    curl -fsSL "https://github.com/Kitware/CMake/releases/download/v${CMAKE_VERSION}/cmake-${CMAKE_VERSION}.tar.gz" | tar -x
    pushd "cmake-${CMAKE_VERSION}/"
    ./configure --prefix="${PREFIX}" --no-system-libs --parallel=${NPROC} --no-qt-gui
    make -j${NPROC}
    make install
    popd
    rm -rf cmake-${CMAKE_VERSION}/
fi

echo "Checking CMake installation:"
cmake --version | grep -F ${CMAKE_VERSION}

if [ -f "${PREFIX}/include/omp.h" -a -f "${PREFIX}/lib/libomp.a" -a -f "${PREFIX}/lib/libomp.dylib" ]; then
    echo "libomp already installed"
else
    echo "Installing libomp ${LIBOMP_VERSION}"
    curl -fsSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LIBOMP_VERSION}/openmp-${LIBOMP_VERSION}.src.tar.xz" | tar -x
    pushd "openmp-${LIBOMP_VERSION}.src/"
    curl -fsSL "https://github.com/llvm/llvm-project/releases/download/llvmorg-${LIBOMP_VERSION}/cmake-${LIBOMP_VERSION}.src.tar.xz" | tar -x -C cmake/ --strip-components=2
    cmake -S . -B build/shared -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DLIBOMP_INSTALL_ALIASES=OFF
    cmake -S . -B build/static -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DLIBOMP_INSTALL_ALIASES=OFF -DLIBOMP_ENABLE_SHARED=OFF
    for build in shared static; do
        cmake --build build/${build}
        cmake --install build/${build}
    done
    popd
    rm -rf "openmp-${LIBOMP_VERSION}.src/"
fi

if [ -f "${PREFIX}/lib/libfftw3f.a" -a -f "${PREFIX}/lib/libfftw3f.dylib" -a -f "${PREFIX}/include/fftw3.h" ]; then
    echo "FFTW already installed"
else
    echo "Installing FFTW ${FFTW_VERSION}"
    curl -fsSL "https://fftw.org/fftw-${FFTW_VERSION}.tar.gz" | tar -x
    pushd "fftw-${FFTW_VERSION}"
    ./configure --enable-shared --disable-debug --disable-dependency-tracking \
        --enable-threads --disable-mpi --disable-openmp --disable-fortran \
        --enable-neon --enable-single \
        --prefix="${PREFIX}"
    make -j${NPROC}
    make install
    popd
    rm -rf "fftw-${FFTW_VERSION}/"
fi

if [ -f "${PREFIX}/bin/clinfo" ]; then
    echo "clinfo already installed" 
else
    echo "Installing clinfo ${CLINFO_VERSION}"
    curl -fsSL "https://github.com/Oblomov/clinfo/archive/${CLINFO_VERSION}.tar.gz" | tar -x
    pushd "clinfo-${CLINFO_VERSION}/"
    make
    cp ./clinfo "${PREFIX}/bin/"
    popd
    rm -rf "clinfo-${CLINFO_VERSION}/"
fi

if [ -f "${PREFIX}/bin/ccache" ]; then
    echo "Ccache already installed" 
else
    echo "Installing Ccache ${CCACHE_VERSION}"
    curl -fsSL "https://github.com/ccache/ccache/releases/download/v${CCACHE_VERSION}/ccache-${CCACHE_VERSION}.tar.xz" | tar -x
    pushd "ccache-${CCACHE_VERSION}/"
    cmake -S . -B build/ -DCMAKE_INSTALL_PREFIX="${PREFIX}" -DCMAKE_BUILD_TYPE=Release -DENABLE_DOCUMENTATION=OFF -DENABLE_TESTING=OFF -DSTATIC_LINK=ON
    cmake --build build/
    cmake --install build/
    popd
    rm -rf "ccache-${CCACHE_VERSION}/"
fi

echo "Bootstrapping done!"
