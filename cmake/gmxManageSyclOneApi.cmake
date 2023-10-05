#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2020- The GROMACS Authors
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

# CMake issue tracking the efforts to make a universal upstream module:
# https://gitlab.kitware.com/cmake/cmake/-/issues/21711

include(gmxFindFlagsForSource)

if(NOT GMX_GPU_SYCL OR GMX_SYCL_HIPSYCL)
    message(FATAL_ERROR "Internal error: OneAPI configuration script was included when it should not")
endif()

if(WIN32)
    if(CMAKE_VERSION VERSION_LESS "3.23.0")
        message(FATAL_ERROR "SYCL with DPC++ on Windows requires cmake 3.23 or later.")
    endif()
    if(NOT BUILD_SHARED_LIBS)
        message(FATAL_ERROR "SYCL with DPC++ on Windows doesn't work with static libraries. Set BUILD_SHARED_LIBS=on.")
        # Tested up to 3.23.1 and icx 2022.1. Problem is order of exe link argument order. Works if gromacs.lib
        # and -fsycl both appear before -link. Not possible to change order from cmake script. cmake fix is WIP.
    endif()
endif()
if(CMAKE_CXX_COMPILER MATCHES "dpcpp")
    message(FATAL_ERROR "Intel's \"dpcpp\" compiler is deprecated; please use \"icpx\" for SYCL builds")
endif()

# Find the flags to enable (or re-enable) SYCL with Intel extensions. In case we turned it off above,
# it's important that we check the combination of both flags, to make sure the second one re-enables SYCL.
if(NOT CHECK_SYCL_CXX_FLAGS_QUIETLY)
    message(STATUS "Checking for flags to enable SYCL")
endif()
set(SAMPLE_SYCL_SOURCE
    "#include <CL/sycl.hpp>
         int main(){
             sycl::queue q(sycl::default_selector{});
             return 0;
         }")
set(SYCL_CXX_FLAGS "-fsycl")
gmx_check_source_compiles_with_flags(
    "${SAMPLE_SYCL_SOURCE}"
    "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_CXX_FLAGS}"
    "CXX"
    SYCL_CXX_FLAGS_RESULT
)
if (SYCL_CXX_FLAGS_RESULT)
    if(NOT CHECK_SYCL_CXX_FLAGS_QUIETLY)
        message(STATUS "Checking for flags to enable SYCL - ${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_CXX_FLAGS}")
    endif()
    set(CHECK_SYCL_CXX_FLAGS_QUIETLY 1 CACHE INTERNAL "Keep quiet on future calls to detect SYCL flags" FORCE)
    set(SYCL_TOOLCHAIN_CXX_FLAGS "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_CXX_FLAGS}")
    set(SYCL_TOOLCHAIN_LINKER_FLAGS "${SYCL_TOOLCHAIN_LINKER_FLAGS} ${SYCL_CXX_FLAGS}")
else()
    message(FATAL_ERROR "Cannot compile a SYCL program with ${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_CXX_FLAGS}. Try a different compiler or disable SYCL.")
endif()

# Add kernel-splitting flag if available, both for compiling and linking
set(SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS "-fsycl-device-code-split=per_kernel")
gmx_check_source_compiles_with_flags(
    "${SAMPLE_SYCL_SOURCE}"
    "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS}"
    "CXX"
    SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS_RESULT
)
if (SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS_RESULT)
    set(SYCL_TOOLCHAIN_CXX_FLAGS "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS}")
    set(SYCL_TOOLCHAIN_LINKER_FLAGS "${SYCL_TOOLCHAIN_LINKER_FLAGS} ${SYCL_DEVICE_CODE_SPLIT_CXX_FLAGS}")
else()
    message(WARNING "Cannot compile SYCL with per-kernel device-code splitting. Simulations will work, but the first step will be much slower than it needs to be. Try a different compiler.")
endif()

# Add fast-math flag where available
gmx_find_flag_for_source(
    SYCL_FAST_MATH_CXX_FLAGS_RESULT
    "${SAMPLE_SYCL_SOURCE}"
    "CXX"
    SYCL_TOOLCHAIN_CXX_FLAGS
    SYCL_FAST_MATH_CXX_FLAGS
    "-ffast-math" "/clang:-ffast-math")
if (SYCL_FAST_MATH_CXX_FLAGS_RESULT)
    set(SYCL_TOOLCHAIN_CXX_FLAGS "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_FAST_MATH_CXX_FLAGS}")
endif()

if("${SYCL_CXX_FLAGS_EXTRA}" MATCHES "fsycl-targets=.*(nvptx64|amdgcn)")
    # When compiling for NVIDIA/AMD, Intel LLVM produces tons of harmless warnings, ignore them
    set(SYCL_WARNINGS_CXX_FLAGS "-Wno-linker-warnings -Wno-override-module -Wno-sycl-target")
    gmx_check_source_compiles_with_flags(
        "${SAMPLE_SYCL_SOURCE}"
        "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_WARNING_CXX_FLAGS}"
        "CXX"
        SYCL_WARNINGS_CXX_FLAGS_RESULT
    )
    if (SYCL_WARNINGS_CXX_FLAGS_RESULT)
        set(SYCL_TOOLCHAIN_CXX_FLAGS "${SYCL_TOOLCHAIN_CXX_FLAGS} ${SYCL_WARNINGS_CXX_FLAGS}")
        set(SYCL_TOOLCHAIN_LINKER_FLAGS "${SYCL_TOOLCHAIN_LINKER_FLAGS} ${SYCL_WARNINGS_CXX_FLAGS}")
    endif()
endif()

if(GMX_GPU_FFT_VKFFT)
    include(gmxManageVkFft)
    if ("${SYCL_CXX_FLAGS_EXTRA}" MATCHES "fsycl-targets=.*nvptx64")
        gmx_manage_vkfft("CUDA")
    elseif ("${SYCL_CXX_FLAGS_EXTRA}" MATCHES "fsycl-targets=.*amdgcn")
        gmx_manage_vkfft("HIP")
    else()
        message(FATAL_ERROR "VkFFT can only be used with CUDA or HIP backend")
    endif()
    set(_sycl_has_valid_fft TRUE)
endif()

include(gmxManageFFTLibraries)

if(GMX_GPU_FFT_MKL)
    #MKLROOT is set by gmxManageFFTLibraries.cmake
    find_library(mkl_sycl_PATH mkl_sycl PATHS "${MKLROOT}/lib" "${MKLROOT}/lib/intel64" REQUIRED)
    mark_as_advanced(mkl_sycl_PATH)
    list(APPEND GMX_EXTRA_LIBRARIES "${mkl_sycl_PATH};OpenCL")

    set(CMAKE_REQUIRED_FLAGS "${SYCL_TOOLCHAIN_CXX_FLAGS}")
    set(CMAKE_REQUIRED_LIBRARIES "${GMX_EXTRA_LIBRARIES};${FFT_LIBRARIES}")
    check_cxx_source_compiles("
#include <oneapi/mkl/dfti.hpp>
int main() {
    oneapi::mkl::dft::descriptor<oneapi::mkl::dft::precision::SINGLE, oneapi::mkl::dft::domain::REAL> d({3,5,7});
    sycl::queue q;
    d.commit(q);
}"
        CAN_LINK_SYCL_MKL)
    unset(CMAKE_REQUIRED_FLAGS)
    unset(CMAKE_REQUIRED_LIBRARIES)
    unset(CMAKE_REQUIRED_INCLUDES)
    if (NOT CAN_LINK_SYCL_MKL)
        message(WARNING "Cannot link mkl_sycl. Make sure the MKL and compiler versions are compatible.")
    endif()

    set(_sycl_has_valid_fft TRUE)
endif()

if(GMX_GPU_FFT_BBFFT)
    # The double-batched FFT library is still called by its former
    # name bbfft in the implementation. For now, only the shared
    # libraries can link into GROMACS shared libraries.
    if (BUILD_SHARED_LIBS)
        find_package(bbfft-sycl 0.3.1 REQUIRED shared)
    else()
        find_package(bbfft-sycl 0.3.1 REQUIRED)
    endif()
    set(_sycl_has_valid_fft TRUE)
endif()

# convert the space-separated strings to lists
separate_arguments(SYCL_TOOLCHAIN_CXX_FLAGS)
list(APPEND SYCL_TOOLCHAIN_CXX_FLAGS ${SYCL_CXX_FLAGS_EXTRA})
separate_arguments(SYCL_TOOLCHAIN_LINKER_FLAGS)
list(APPEND SYCL_TOOLCHAIN_LINKER_FLAGS ${SYCL_CXX_FLAGS_EXTRA})

# Make strings for pretty-printing in gmx -version
string(REPLACE ";" " " SYCL_TOOLCHAIN_CXX_FLAGS_STR "${SYCL_TOOLCHAIN_CXX_FLAGS}")
string(STRIP "${SYCL_TOOLCHAIN_CXX_FLAGS_STR}" SYCL_TOOLCHAIN_CXX_FLAGS_STR)
string(REPLACE ";" " " SYCL_TOOLCHAIN_LINKER_FLAGS_STR "${SYCL_TOOLCHAIN_LINKER_FLAGS}")
string(STRIP "${SYCL_TOOLCHAIN_LINKER_FLAGS_STR}" SYCL_TOOLCHAIN_LINKER_FLAGS_STR)

# Add function wrapper similar to the one used by ComputeCPP and hipSYCL
function(add_sycl_to_target)
    cmake_parse_arguments(
        PARSE_ARGV 0 # No positional arguments
        ARGS # Prefix for the resulting variables
        "" # No options
        "TARGET" # One-value keyword
        "SOURCES" # Multi-value keyword
    )
    set_property(SOURCE ${ARGS_SOURCES} APPEND PROPERTY COMPILE_OPTIONS ${SYCL_TOOLCHAIN_CXX_FLAGS})
    target_link_options(${ARGS_TARGET} PRIVATE ${SYCL_TOOLCHAIN_LINKER_FLAGS})
endfunction(add_sycl_to_target)
