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

# This test is slow with OpenCL or the OpenCL DPC++ SYCL back end so
# label it as SLOW_TEST always. However when both those cases have
# gone, it no longer needs this annotation.
set(exename "fft-test")
set(sources
    clfftinitializer.cpp
    )
set(sources_needing_sycl
    fft.cpp
    )
if (GMX_GPU_CUDA)
    set(CUDA_LINK_LIBRARIES_KEYWORD PRIVATE)
    cuda_add_executable(${exename} ${sources} ${sources_needing_sycl)
    target_link_libraries(${exename} PRIVATE libgromacs)
else()
    add_executable(${exename} ${sources} ${sources_needing_sycl})
    if (GMX_GPU_SYCL)
        add_sycl_to_target(
            TARGET ${exename}
            SOURCES ${sources_needing_sycl}
            )
    endif()
endif()

gmx_target_compile_options(${exename})
target_compile_definitions(${exename} PRIVATE HAVE_CONFIG_H)
target_compile_definitions(${exename} PRIVATE TEST_USES_HARDWARE_DETECTION=true)
target_link_libraries(${exename} PRIVATE
                      testutils
                      common
                      fft
                      gmock
                      )
gmx_warn_on_everything(${exename})
if (HAS_WARNING_EVERYTHING)
    # Some false positives exist produced by GoogleTest implementation
    gmx_target_warning_suppression(${exename} "-Wno-zero-as-null-pointer-constant" HAS_WARNING_NO_ZERO_AS_NULL_POINTER_CONSTANT)
    gmx_target_warning_suppression(${exename} "-Wno-gnu-zero-variadic-macro-arguments" HAS_WARNING_NO_GNU_ZERO_VARIADIC_MACRO_ARGUMENTS)
    # Use of GoogleMock can generate mock member functions that are unused
    gmx_target_warning_suppression(${exename} "-Wno-unused-member-function" HAS_WARNING_NO_UNUSED_MEMBER_FUNCTION)
    if(GMX_GPU_CUDA)
        # CUDA headers target C, so use old-style casts that clang
        # warns about when it is the host compiler
        gmx_target_warning_suppression(${exename} "-Wno-old-style-cast" HAS_NO_OLD_STYLE_CAST)
    endif()
endif()

gmx_register_gtest_test(FFTUnitTests ${exename} SLOW_TEST QUICK_GPU_TEST)

if(GMX_USE_Heffte OR GMX_USE_cuFFTMp)
    gmx_add_mpi_unit_test(FFTMpiUnitTests fft-mpi-test 4 HARDWARE_DETECTION
        GPU_CPP_SOURCE_FILES
        fft_mpi.cpp
        )
endif()
