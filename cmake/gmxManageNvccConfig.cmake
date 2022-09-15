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

# Manage CUDA nvcc compilation configuration, try to be smart to ease the users'
# pain as much as possible:
# - use the CUDA_HOST_COMPILER if defined by the user, otherwise
# - check if nvcc works with CUDA_HOST_COMPILER and the generated nvcc and C++ flags
#
# - (advanced) variables set:
#   * CUDA_HOST_COMPILER_OPTIONS    - the full host-compiler related option list passed to nvcc
#
# Note that from CMake 2.8.10 FindCUDA defines CUDA_HOST_COMPILER internally,
# so we won't set it ourselves, but hope that the module does a good job.

# glibc 2.23 changed string.h in a way that breaks CUDA compilation in
# many projects, but which has a trivial workaround. It would be nicer
# to compile with nvcc and see that the workaround is necessary and
# effective, but it is unclear how to do that. Also, grepping in the
# glibc source shows that _FORCE_INLINES is only used in this string.h
# feature and performance of memcpy variants is unimportant for CUDA
# code in GROMACS. So this workaround is good enough to keep problems
# away from users installing GROMACS. See Issue #1982.
function(work_around_glibc_2_23)
    try_compile(IS_GLIBC_2_23_OR_HIGHER ${CMAKE_BINARY_DIR} ${CMAKE_SOURCE_DIR}/cmake/TestGlibcVersion.cpp)
    if(IS_GLIBC_2_23_OR_HIGHER)
        message(STATUS "Adding work-around for issue compiling CUDA code with glibc 2.23 string.h")
        list(APPEND CUDA_HOST_COMPILER_OPTIONS "-D_FORCE_INLINES")
        set(CUDA_HOST_COMPILER_OPTIONS ${CUDA_HOST_COMPILER_OPTIONS} PARENT_SCOPE)
    endif()
endfunction()

gmx_check_if_changed(CUDA_HOST_COMPILER_CHANGED CUDA_HOST_COMPILER)

# set up host compiler and its options
if(CUDA_HOST_COMPILER_CHANGED)
    set(CUDA_HOST_COMPILER_OPTIONS "")

    if(APPLE AND CMAKE_C_COMPILER_ID MATCHES "GNU")
        # Some versions of gcc-4.8 and gcc-4.9 have produced errors
        # (in particular on OS X) if we do not use
        # -D__STRICT_ANSI__. It is harmless, so we might as well add
        # it for all versions.
        list(APPEND CUDA_HOST_COMPILER_OPTIONS "-D__STRICT_ANSI__")
    endif()

    work_around_glibc_2_23()

    set(CUDA_HOST_COMPILER_OPTIONS "${CUDA_HOST_COMPILER_OPTIONS}"
        CACHE STRING "Options for nvcc host compiler (do not edit!).")

    mark_as_advanced(CUDA_HOST_COMPILER CUDA_HOST_COMPILER_OPTIONS)
endif()


# Tests a single set of one or more flags to use with nvcc.
#
# If the flags are accepted, they are appended to the variable named
# in the first argument. The cache variable named in the second
# argument is used to avoid rerunning the check in future invocations
# of cmake. The list of flags to check follows these two required
# arguments.
#
# As this code is not yet tested on Windows, it always accepts the
# flags in that case.
function(gmx_add_nvcc_flag_if_supported _output_variable_name_to_append_to _flags_cache_variable_name _set_as_shell_flag)
    # If the check has already been run, do not re-run it
    if (NOT ${_flags_cache_variable_name} AND NOT WIN32)
        message(STATUS "Checking if nvcc accepts flags ${ARGN}")
        execute_process(
            COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${ARGN} -ccbin ${CUDA_HOST_COMPILER} "${CMAKE_SOURCE_DIR}/cmake/TestCUDA.cu"
            RESULT_VARIABLE _cuda_success
            OUTPUT_QUIET
            ERROR_QUIET
            )
        # Convert the success value to a boolean and report status
        if (_cuda_success EQUAL 0)
            set(_cache_variable_value TRUE)
            message(STATUS "Checking if nvcc accepts flags ${ARGN} - Success")
        else()
            if(NOT(CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 11))
              set(CCBIN "-ccbin ${CUDA_HOST_COMPILER}")
            endif()
            execute_process(
                COMMAND ${CUDAToolkit_NVCC_EXECUTABLE} ${ARGN} ${CCBIN} "${CMAKE_SOURCE_DIR}/cmake/TestCUDA.cu"
                RESULT_VARIABLE _cuda_success
                OUTPUT_QUIET
                ERROR_QUIET
                )
            # Convert the success value to a boolean and report status
            if (_cuda_success EQUAL 0)
                set(_cache_variable_value TRUE)
                message(STATUS "Checking if nvcc accepts flags ${ARGN} - Success")
            else()
                set(_cache_variable_value FALSE)
                message(STATUS "Checking if nvcc accepts flags ${ARGN} - Failed")
            endif()
        endif()
        set(${_flags_cache_variable_name} ${_cache_variable_value} CACHE BOOL "Whether NVCC supports flag(s) ${ARGN}")
    endif()
    # Append the flags to the output variable if they have been tested to work
    if (${_flags_cache_variable_name} OR WIN32)
      if(${_set_as_shell_flag})
	string(REPLACE ";" " " _flag "${ARGN}")
        list(APPEND ${_output_variable_name_to_append_to} "SHELL:${_flag}")
        set(${_output_variable_name_to_append_to} ${${_output_variable_name_to_append_to}} PARENT_SCOPE)
      else()
	list(APPEND ${_output_variable_name_to_append_to} "${ARGN}")
        set(${_output_variable_name_to_append_to} ${${_output_variable_name_to_append_to}} PARENT_SCOPE)
      endif()
    endif()
endfunction()

# If any of these manual override variables for target CUDA GPU architectures
# or virtual architecture is set, parse the values and assemble the nvcc
# command line for these. Otherwise use our defaults.
# Note that the manual override variables require a semicolon separating
# architecture codes.
set(GMX_CUDA_NVCC_GENCODE_FLAGS)
if (GMX_CUDA_TARGET_SM OR GMX_CUDA_TARGET_COMPUTE)
    set(_target_sm_list ${GMX_CUDA_TARGET_SM})
    foreach(_target ${_target_sm_list})
        gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_${_target} FALSE --generate-code=arch=compute_${_target},code=[compute_${_target},sm_${_target}])
        if (NOT NVCC_HAS_GENCODE_COMPUTE_AND_SM_${_target} AND NOT WIN32)
            message(FATAL_ERROR "Your choice of ${_target} in GMX_CUDA_TARGET_SM was not accepted by nvcc, please choose a target that it accepts")
        endif()
    endforeach()
    set(_target_compute_list ${GMX_CUDA_TARGET_COMPUTE})
    foreach(_target ${_target_compute_list})
        gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_${_target} FALSE --generate-code=arch=compute_${_target},code=[compute_${_target},sm_${_target}])
        if (NOT NVCC_HAS_GENCODE_COMPUTE_${_target} AND NOT WIN32)
            message(FATAL_ERROR "Your choice of ${_target} in GMX_CUDA_TARGET_COMPUTE was not accepted by nvcc, please choose a target that it accepts")
        endif()
    endforeach()
else()
    # Set the CUDA GPU architectures to compile for:
    # - with CUDA >=11.0        CC 8.0 is supported
    #     => compile sm_35, sm_37, sm_50, sm_52, sm_60, sm_61, sm_70, sm_75, sm_80 SASS, and compute_35, compute_80 PTX

    # First add flags that trigger SASS (binary) code generation for physical arch
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_35 FALSE --generate-code=arch=compute_35,code=[compute_35,sm_35])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_37 FALSE --generate-code=arch=compute_37,code=[compute_37,sm_37])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_50 FALSE --generate-code=arch=compute_50,code=[compute_50,sm_50])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_52 FALSE --generate-code=arch=compute_52,code=[compute_52,sm_52])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_60 FALSE --generate-code=arch=compute_60,code=[compute_60,sm_60])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_61 FALSE --generate-code=arch=compute_61,code=[compute_61,sm_61])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_70 FALSE --generate-code=arch=compute_70,code=[compute_70,sm_70])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_75 FALSE --generate-code=arch=compute_75,code=[compute_75,sm_75])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_80 FALSE --generate-code=arch=compute_80,code=[compute_80,sm_80])
    # We use this to avoid a warning in our GitLab CI with a gcc 7 base image (see above for details)
    if (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 8 AND NOT DEFINED ENV{GITLAB_CI})
        gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_AND_SM_86 FALSE --generate-code=arch=compute_86,code=[compute_86,sm_86])
    endif()
    # Requesting sm or compute 35, 37, or 50 triggers deprecation messages with
    # nvcc 11.0, which we need to suppress for use in CI
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_WARNING_NO_DEPRECATED_GPU_TARGETS FALSE --Wno-deprecated-gpu-targets)

    # Next add flags that trigger PTX code generation for the
    # newest supported virtual arch that's useful to JIT to future architectures
    # as well as an older one suitable for JIT-ing to any rare intermediate arch
    # (like that of Jetson / Drive PX devices)
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_53 FALSE --generate-code=arch=compute_53,code=[compute_53,sm_53])
    gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_GENCODE_FLAGS NVCC_HAS_GENCODE_COMPUTE_80 FALSE --generate-code=arch=compute_80,code=[compute_80,sm_80])
endif()

if (GMX_CUDA_TARGET_SM)
    set_property(CACHE GMX_CUDA_TARGET_SM PROPERTY HELPSTRING "List of CUDA GPU architecture codes to compile for (without the sm_ prefix)")
    set_property(CACHE GMX_CUDA_TARGET_SM PROPERTY TYPE STRING)
endif()
if (GMX_CUDA_TARGET_COMPUTE)
    set_property(CACHE GMX_CUDA_TARGET_COMPUTE PROPERTY HELPSTRING "List of CUDA virtual architecture codes to compile for (without the compute_ prefix)")
    set_property(CACHE GMX_CUDA_TARGET_COMPUTE PROPERTY TYPE STRING)
endif()

# assemble the CUDA flags
list(APPEND GMX_CUDA_NVCC_FLAGS ${GMX_CUDA_NVCC_GENCODE_FLAGS})

gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_FLAGS NVCC_HAS_USE_FAST_MATH FALSE --use_fast_math)
# Add warnings
gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_FLAGS NVCC_HAS_PTXAS_WARN_DOUBLE_USAGE TRUE --ptxas-options -warn-double-usage)
gmx_add_nvcc_flag_if_supported(GMX_CUDA_NVCC_FLAGS NVCC_HAS_PTXAS_WERROR TRUE --ptxas-options -Werror)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # CUDA header cuda_runtime_api.h in at least CUDA 10.1 uses 0
    # where nullptr would be preferable. GROMACS can't fix these, so
    # must suppress them.
    GMX_TEST_CXXFLAG(HAS_WARNING_NO_ZERO_AS_NULL_POINTER_CONSTANT "-Wno-zero-as-null-pointer-constant" NVCC_CLANG_SUPPRESSIONS_CXXFLAGS)

    # CUDA header crt/math_functions.h in at least CUDA 10.x and 11.1
    # used throw() specifications that are deprecated in more recent
    # C++ versions. GROMACS can't fix these, so must suppress them.
    GMX_TEST_CXXFLAG(HAS_WARNING_NO_DEPRECATED_DYNAMIC_EXCEPTION_SPEC "-Wno-deprecated-dynamic-exception-spec" NVCC_CLANG_SUPPRESSIONS_CXXFLAGS)

    # CUDA headers cuda_runtime.h and channel_descriptor.h in at least
    # CUDA 11.0 uses many C-style casts, which are ncessary for this
    # header to work for C. GROMACS can't fix these, so must suppress
    # the warnings they generate
    GMX_TEST_CXXFLAG(HAS_WARNING_NO_OLD_STYLE_CAST "-Wno-old-style-cast" NVCC_CLANG_SUPPRESSIONS_CXXFLAGS)

    # Add these flags to those used for the host compiler. The
    # "-Xcompiler" prefix directs nvcc to only use them for host
    # compilation, which is all that is needed in this case.
    foreach(_flag ${NVCC_CLANG_SUPPRESSIONS_CXXFLAGS})
        list(APPEND GMX_CUDA_NVCC_FLAGS "-Xcompiler ${_flag}")
    endforeach()
endif()

string(TOUPPER "${CMAKE_BUILD_TYPE}" _build_type)
gmx_check_if_changed(_cuda_nvcc_executable_or_flags_changed CUDA_NVCC_EXECUTABLE CUDA_NVCC_FLAGS CUDA_NVCC_FLAGS_${_build_type})
