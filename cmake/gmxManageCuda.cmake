#
# This file is part of the GROMACS molecular simulation package.
#
# Copyright 2012- The GROMACS Authors
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

set(GMX_GPU_CUDA ON)

option(GMX_CLANG_CUDA "Use clang for CUDA" OFF)

if(GMX_DOUBLE)
    message(FATAL_ERROR "CUDA acceleration is not available in double precision")
endif()

if(GMX_CLANG_CUDA)
    include(gmxManageClangCudaConfig)
    list(APPEND GMX_EXTRA_LIBRARIES ${GMX_CUDA_CLANG_LINK_LIBS})
    link_directories("${GMX_CUDA_CLANG_LINK_DIRS}")
else()
    # Using NVIDIA compiler
    if(NOT CUDAToolkit_NVCC_EXECUTABLE)
        message(FATAL_ERROR "nvcc is required for a CUDA build, please set CUDA_TOOLKIT_ROOT_DIR appropriately")
    endif()
    # set up nvcc options
    include(gmxManageNvccConfig)
endif()

option(GMX_CUDA_NB_SINGLE_COMPILATION_UNIT "Whether to compile the CUDA non-bonded module using a single compilation unit." OFF)
mark_as_advanced(GMX_CUDA_NB_SINGLE_COMPILATION_UNIT)
