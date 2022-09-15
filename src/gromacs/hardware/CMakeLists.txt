#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2015- The GROMACS Authors
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

add_library(hardware OBJECT)
target_sources(hardware PRIVATE
    cpuinfo.cpp
    hardwaretopology.cpp
    printhardware.cpp
    identifyavx512fmaunits.cpp
    simd_support.cpp
    )
if (BUILD_SHARED_LIBS)
    set_target_properties(hardware PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

# We would like to have device_management_common.cpp and detecthardware.cpp in the target_sources call abaove.
# However, we instead put them in each branch below because they need to include device_information.h, which
# in the case of SYCL means that they have to be in an add_sycl_to_target call to pick up <CL/sycl.hpp>.
if(GMX_GPU_OPENCL)
    target_sources(hardware PRIVATE
            device_management.cpp
            detecthardware.cpp
            device_management_common.cpp
            device_management_ocl.cpp
        )
elseif(GMX_GPU_CUDA)
    target_sources(hardware PRIVATE
            device_management.cu
        )
    _gmx_add_files_to_property(CUDA_SOURCES
        # Must add these files so they can include device_information.h
        device_management_common.cpp
        detecthardware.cpp
	)
    target_link_libraries(hardware PRIVATE CUDA::cudart)
elseif(GMX_GPU_SYCL)
    add_sycl_to_target(TARGET hardware
            device_management_sycl.cpp
            device_management_sycl_intel_device_ids.cpp
            device_management.cpp
            device_management_common.cpp
            detecthardware.cpp
            )
else()
    target_sources(hardware PRIVATE
            device_management.cpp
            detecthardware.cpp
            device_management_common.cpp
    )
endif()

if(SIMD_AVX_512_CXX_SUPPORTED AND NOT ("${GMX_SIMD_ACTIVE}" STREQUAL "AVX_512_KNL"))
    # Since we might be overriding -march=core-avx2, add a flag so we don't warn for this specific file.
    # On KNL this can cause illegal instruction because the compiler might use non KNL AVX instructions
    # with the SIMD_AVX_512_CXX_FLAGS flags.
    set_source_files_properties(identifyavx512fmaunits.cpp PROPERTIES COMPILE_FLAGS "${SIMD_AVX_512_CXX_FLAGS} ${CXX_NO_UNUSED_OPTION_WARNING_FLAGS}")
endif()

# Source files have the following private module dependencies.
target_link_libraries(hardware PRIVATE
                      common
		      gpu_utils
                      simd
                      utility
                      )

# Public interface for modules, including dependencies and interfaces
target_include_directories(hardware PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if (BUILD_TESTING)
    add_subdirectory(tests)
endif()
