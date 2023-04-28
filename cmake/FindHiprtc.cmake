#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2023- The GROMACS Authors
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

# This CMake find package follows conventions, namely it sets Hiprtc_FOUND
# cache variable upon success, and creates the shared library target
# Hiprtc::Hiprtc for other targets to depend on.

include(FindPackageHandleStandardArgs)
find_library(Hiprtc_LIBRARY
    NAMES hiprtc
    PATHS
        "$ENV{HIP_PATH}"
        "$ENV{ROCM_PATH}/hip"
        /opt/rocm/
    PATH_SUFFIXES
        lib
        lib64
    )
find_path(Hiprtc_INCLUDE_DIR
    NAMES hip/hiprtc.h
    PATHS
        "$ENV{HIP_PATH}"
        "$ENV{ROCM_PATH}/hip"
        /opt/rocm/
    PATH_SUFFIXES
        include
        ../include
    )

find_package_handle_standard_args(Hiprtc REQUIRED_VARS Hiprtc_LIBRARY Hiprtc_INCLUDE_DIR)

if (Hiprtc_FOUND)
    mark_as_advanced(Hiprtc_LIBRARY)
    mark_as_advanced(Hiprtc_INCLUDE_DIR)
endif()

if (Hiprtc_FOUND AND NOT TARGET Hiprtc::Hiprtc)
  add_library(Hiprtc::Hiprtc SHARED IMPORTED)
  set_property(TARGET Hiprtc::Hiprtc PROPERTY IMPORTED_LOCATION ${Hiprtc_LIBRARY})
  target_include_directories(Hiprtc::Hiprtc INTERFACE ${Hiprtc_INCLUDE_DIR})
endif()
