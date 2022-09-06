#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2022- The GROMACS Authors
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

# Note, source files should be listed in alphabetical order, to reduce
# the incidence of textual clashes when adding/moving files that
# otherwise make the end of the list a hotspot.

add_library(device_info OBJECT)
target_sources(device_info PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/device_management_common.cpp
  )
if (BUILD_SHARED_LIBS)
    set_target_properties(device_info PROPERTIES POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(device_info PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)

if(GMX_GPU_OPENCL)
    target_sources(device_info PRIVATE
            device_management_ocl.cpp
        )
elseif(GMX_GPU_CUDA)
    target_sources(device_info PRIVATE
            device_management.cu
        )
    _gmx_add_files_to_property(CUDA_SOURCES
        # Must add these files so they can include device_information.h                                                                                                                                                                                                                                                       
        device_management_common.cpp
        detecthardware.cpp
        )
    target_link_libraries(device_info PRIVATE CUDA::cudart)
elseif(GMX_GPU_SYCL)
    target_sources(device_info PRIVATE
            device_management_sycl.cpp
    )
    _gmx_add_files_to_property(SYCL_SOURCES
        device_management_sycl.cpp
        # Must add these files so they can include device_information.h                                                                                                                                                                                                                                                       
        device_management.cpp
        device_management_common.cpp
        detecthardware.cpp
        )
else()
    target_sources(device_info PRIVATE
            device_management.cpp
    )
endif()

target_link_libraries(device_info PRIVATE
  common
  utility
  )
