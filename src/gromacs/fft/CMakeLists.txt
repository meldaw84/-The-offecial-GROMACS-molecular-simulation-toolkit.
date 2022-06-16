#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2013- The GROMACS Authors
# and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
# Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
#
# GROMACS is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public License
# as published by the Free Software Foundation; either version 2.1
# of the License, or (at your option) any later version.
#
# GROMACS is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with GROMACS; if not, see
# https://www.gnu.org/licenses, or write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
#
# If you want to redistribute modifications to GROMACS, please
# consider that scientific software is very special. Version
# control is crucial - bugs must be traceable. We will be happy to
# consider code for inclusion in the official distribution, but
# derived work must not be called official GROMACS. Details are found
# in the README & COPYING files - if they are missing, get the
# official version at https://www.gromacs.org.
#
# To help us fund GROMACS development, we humbly ask that you cite
# the research papers on the package. Check out https://www.gromacs.org.

add_library(fft OBJECT
            calcgrid.cpp
            clfftinitializer.cpp
            fft.cpp
            fft5d.cpp
            parallel_3dfft.cpp
            )

if (BUILD_SHARED_LIBS)
    set_target_properties(fft PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# Handle CUDA separately for now, because FindCUDA.cmake cannot do
# object libraries
if (GMX_GPU_CUDA)
    if(GMX_USE_cuFFTMp)
        gmx_add_libgromacs_sources(
            gpu_3dfft_cufftmp.cpp
            )
    endif()
    gmx_add_libgromacs_sources(
        # CUDA-specific sources
        gpu_3dfft_cufft.cu
        gpu_3dfft_impl.cpp
        gpu_3dfft.cpp
        )
    _gmx_add_files_to_property(CUDA_SOURCES
        # Must add these files so they can include cuda_runtime.h
        gpu_3dfft.cpp
        gpu_3dfft_impl.cpp
        )
    if(GMX_USE_cuFFTMp)
        _gmx_add_files_to_property(CUDA_SOURCES
            # Must add this files as it requires path set in cuda_include_directories
            gpu_3dfft_cufftmp.cpp
        )
    endif()
else()
    target_sources(fft PRIVATE
                   gpu_3dfft.cpp
                   )
endif()

option(GMX_EXTERNAL_CLFFT "True if an external clFFT is required to be used" FALSE)
mark_as_advanced(GMX_EXTERNAL_CLFFT)

if (GMX_GPU_OPENCL)
    target_sources(fft PRIVATE
                   gpu_3dfft_impl.cpp
                   gpu_3dfft_ocl.cpp
                   )
    target_link_libraries(fft PRIVATE ${OpenCL_LIBRARIES})
    set_source_files_properties(gpu_3dfft_ocl.cpp COMPILE_DEFINITIONS ${OpenCL_DEFINITIONS})

    # Default to using clFFT found on the system
    # switch to quiet at the second run.
    if (DEFINED clFFT_LIBRARY)
        set (clFFT_FIND_QUIETLY TRUE)
    endif()
    find_package(clFFT)
    if (NOT clFFT_FOUND)
        if (GMX_EXTERNAL_CLFFT)
            message(FATAL_ERROR "Did not find required external clFFT library, consider setting clFFT_ROOT_DIR")
        endif()

        if(MSVC)
            message(FATAL_ERROR
"An OpenCL build was requested with Visual Studio compiler, but GROMACS
requires clFFT, which was not found on your system. GROMACS does bundle
clFFT to help with building for OpenCL, but that clFFT has not yet been
ported to the more recent versions of that compiler that GROMACS itself
requires. Thus for now, OpenCL is not available with MSVC and the internal
build of clFFT in GROMACS 2019. Either change compiler, try installing
a clFFT package, or use the latest GROMACS 2018 point release.")
        endif()

        # Fall back on the internal version
        set (_clFFT_dir ${PROJECT_SOURCE_DIR}/src/external/clFFT/src)
        add_subdirectory(${_clFFT_dir} clFFT-build)
        # Get usage requirements from clFFT
        target_link_libraries(fft PUBLIC clFFT)
        # Get the actual object files from clFFT also, since they
        # cannot come from the call to target_link_libraries()
        target_sources(fft PUBLIC
            $<TARGET_OBJECTS:clFFT>
        )
        # Use the magic variable for how to link any library needed for
        # dlopen, etc.  which is -ldl where needed, and empty otherwise
        # (e.g. Windows, BSD, Mac).
        target_link_libraries(fft PUBLIC "${CMAKE_DL_LIBS}")
    else()
        target_link_libraries(fft PUBLIC clFFT)
    endif()
endif()

if (GMX_GPU_SYCL)
    if (GMX_SYCL_HIPSYCL)
        if (GMX_HIPSYCL_HAVE_HIP_TARGET)
            target_sources(fft PRIVATE gpu_3dfft_sycl_rocfft.cpp)
            add_sycl_to_target(TARGET fft SOURCES gpu_3dfft_sycl_rocfft.cpp)
        endif()
    else()
        # Must be dpcpp
        if (GMX_FFT_MKL)
            target_sources(fft PRIVATE gpu_3dfft_sycl_mkl.cpp)
            add_sycl_to_target(TARGET fft SOURCES
                               gpu_3dfft_sycl_mkl.cpp
                               gpu_3dfft.cpp
                               gpu_3dfft_impl.cpp
                               gpu_3dfft_sycl.cpp
                               )
        endif()
    endif()
    # Add the source files that don't need special SYCL handling
    target_sources(fft PRIVATE
                   gpu_3dfft_impl.cpp
                   gpu_3dfft_sycl.cpp
                   )
endif()

if (GMX_FFT_FFTPACK)
    target_sources(fft PRIVATE
                   fft_fftpack.cpp
                   ${CMAKE_SOURCE_DIR}/src/external/fftpack/fftpack.cpp
                   )
endif()
if (GMX_FFT_FFTW3 OR GMX_FFT_ARMPL_FFTW3)
    target_sources(fft PRIVATE fft_fftw3.cpp)
endif()
if (GMX_FFT_MKL)
    target_sources(fft PRIVATE fft_mkl.cpp)
endif()

# Source files have the following private module dependencies.
target_link_libraries(fft PRIVATE
                      gmxlib
                      math
                      mdtypes
                      tng_io
                      )

# Public interface for modules, including dependencies and interfaces
target_include_directories(fft PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(fft PUBLIC
                      legacy_api
                      )

target_link_libraries(fft PUBLIC legacy_api)
target_link_libraries(fft PRIVATE common)

# Module dependencies
# This module convey transitive dependence on these modules.
#target_link_libraries(fft PUBLIC
target_link_libraries(fft INTERFACE
                      utility
                      )
# Source files have the following private module dependencies.
target_link_libraries(fft PRIVATE tng_io)
target_link_libraries(fft PRIVATE legacy_modules)

if (GMX_FFT_FFTW3 OR GMX_FFT_ARMPL_FFTW3)
    target_link_libraries(fft PRIVATE ${${FFTW}_LIBRARIES})
endif()

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
