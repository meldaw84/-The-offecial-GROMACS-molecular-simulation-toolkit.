/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2014- The GROMACS Authors
 * and the project initiators Erik Lindahl, Berk Hess and David van der Spoel.
 * Consult the AUTHORS/COPYING files and https://www.gromacs.org for details.
 *
 * GROMACS is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public License
 * as published by the Free Software Foundation; either version 2.1
 * of the License, or (at your option) any later version.
 *
 * GROMACS is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with GROMACS; if not, see
 * https://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at https://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out https://www.gromacs.org.
 */

#ifndef GMX_SIMD_IMPL_RISCV_V_UTIL_FLOAT_H
#define GMX_SIMD_IMPL_RISCV_V_UTIL_FLOAT_H

/*! \libinternal \file
 *
 * \brief RISC-V vector impl., higher-level single prec. SIMD utility functions
 *
 * \author Berk Hess <hess@kth.se>
 *
 * \ingroup module_simd
 */

/* Avoid adding dependencies on the rest of GROMACS here (e.g. gmxassert.h)
 * since we want to be able run the low-level SIMD implementations independently
 * in simulators for new hardware.
 */

#include "config.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include <algorithm>

#include "gromacs/simd/impl_riscv_v/impl_riscv_v_definitions.h"
#include "gromacs/simd/impl_riscv_v/impl_riscv_v_simd_float.h"

namespace gmx
{

/*! \cond libapi */
/*! \addtogroup module_simd */
/*! \{ */

/*! \name Higher-level SIMD utility functions, single precision.
 *
 * These include generic functions to work with triplets of data, typically
 * coordinates, and a few utility functions to load and update data in the
 * nonbonded kernels.
 * These functions should be available on all implementations, although
 * some wide SIMD implementations (width>=8) also provide special optional
 * versions to work with half or quarter registers to improve the performance
 * in the nonbonded kernels.
 *
 * \{
 */

/*! \brief Load 4 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets,
 *         and transpose into 4 SIMD float variables.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 4 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Pointer to the start of the memory area
 * \param      offset Array with offsets to the start of each data point.
 * \param[out] v0     1st component of data, base[align*offset[i]] for each i.
 * \param[out] v1     2nd component of data, base[align*offset[i] + 1] for each i.
 * \param[out] v2     3rd component of data, base[align*offset[i] + 2] for each i.
 * \param[out] v3     4th component of data, base[align*offset[i] + 3] for each i.
 *
 * The floating-point memory locations must be aligned, but only to the smaller
 * of four elements and the floating-point SIMD width.
 *
 * The offset memory must be aligned to GMX_SIMD_DINT32_WIDTH.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 */
template<int align>
static inline void gmx_simdcall gatherLoadTranspose(const float*       base,
                                                    const std::int32_t offset[],
                                                    SimdFloat*         v0,
                                                    SimdFloat*         v1,
                                                    SimdFloat*         v2,
                                                    SimdFloat*         v3)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);
    // Base pointer must be aligned to the smaller of 4 elements and float SIMD width
    assert(std::size_t(base) % (std::min(GMX_SIMD_FLOAT_WIDTH, 4) * sizeof(float)) == 0);
    // align parameter must also be a multiple of the above alignment requirement
    assert(align % std::min(GMX_SIMD_FLOAT_WIDTH, 4) == 0);

    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset[i]];
        v1->simdInternal_[i] = base[align * offset[i] + 1];
        v2->simdInternal_[i] = base[align * offset[i] + 2];
        v3->simdInternal_[i] = base[align * offset[i] + 3];
    }
}

/*! \brief Load 2 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets,
 *         and transpose into 2 SIMD float variables.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 2 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Pointer to the start of the memory area
 * \param      offset Array with offsets to the start of each data point.
 * \param[out] v0     1st component of data, base[align*offset[i]] for each i.
 * \param[out] v1     2nd component of data, base[align*offset[i] + 1] for each i.
 *
 * The floating-point memory locations must be aligned, but only to the smaller
 * of two elements and the floating-point SIMD width.
 *
 * The offset memory must be aligned to GMX_SIMD_FINT32_WIDTH.
 *
 * To achieve the best possible performance, you should store your data with
 * alignment \ref c_simdBestPairAlignmentFloat in single, or
 * \ref c_simdBestPairAlignmentDouble in double.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 */
template<int align>
static inline void gmx_simdcall
gatherLoadTranspose(const float* base, const std::int32_t offset[], SimdFloat* v0, SimdFloat* v1)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);
    // Base pointer must be aligned to the smaller of 2 elements and float SIMD width
    assert(std::size_t(base) % (std::min(GMX_SIMD_FLOAT_WIDTH, 2) * sizeof(float)) == 0);
    // align parameter must also be a multiple of the above alignment requirement
    assert(align % std::min(GMX_SIMD_FLOAT_WIDTH, 2) == 0);

    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset[i]];
        v1->simdInternal_[i] = base[align * offset[i] + 1];
    }
}


/*! \brief Best alignment to use for aligned pairs of float data.
 *
 *  The routines to load and transpose data will work with a wide range of
 *  alignments, but some might be faster than others, depending on the load
 *  instructions available in the hardware. This specifies the best
 *  alignment for each implementation when working with pairs of data.
 *
 *  To allow each architecture to use the most optimal form, we use a constant
 *  that code outside the SIMD module should use to store things properly. It
 *  must be at least 2. For example, a value of 2 means the two parameters A & B
 *  are stored as [A0 B0 A1 B1] while align-4 means [A0 B0 - - A1 B1 - -].
 *
 *  This alignment depends on the efficiency of partial-register load/store
 *  operations, and will depend on the architecture.
 */
static const int c_simdBestPairAlignmentFloat = 2;


/*! \brief Load 3 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets,
 *         and transpose into 3 SIMD float variables.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 3 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Pointer to the start of the memory area
 * \param      offset Array with offsets to the start of each data point.
 * \param[out] v0     1st component of data, base[align*offset[i]] for each i.
 * \param[out] v1     2nd component of data, base[align*offset[i] + 1] for each i.
 * \param[out] v2     3rd component of data, base[align*offset[i] + 2] for each i.
 *
 * This function can work with both aligned (better performance) and unaligned
 * memory. When the align parameter is not a power-of-two (align==3 would be normal
 * for packed atomic coordinates) the memory obviously cannot be aligned, and
 * we account for this.
 * However, in the case where align is a power-of-two, we assume the base pointer
 * also has the same alignment, which will enable many platforms to use faster
 * aligned memory load operations.
 * An easy way to think of this is that each triplet of data in memory must be
 * aligned to the align parameter you specify when it's a power-of-two.
 *
 * The offset memory must always be aligned to GMX_SIMD_FINT32_WIDTH, since this
 * enables us to use SIMD loads and gather operations on platforms that support it.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This routine uses a normal array for the offsets, since we typically
 *       load this data from memory. On the architectures we have tested this
 *       is faster even when a SIMD integer datatype is present.
 * \note To improve performance, this function might use full-SIMD-width
 *       unaligned loads. This means you need to ensure the memory is padded
 *       at the end, so we always can load GMX_SIMD_REAL_WIDTH elements
 *       starting at the last offset. If you use the Gromacs aligned memory
 *       allocation routines this will always be the case.
 */
template<int align>
static inline void gmx_simdcall gatherLoadUTranspose(const float*       base,
                                                     const std::int32_t offset[],
                                                     SimdFloat*         v0,
                                                     SimdFloat*         v1,
                                                     SimdFloat*         v2)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);

    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset[i]];
        v1->simdInternal_[i] = base[align * offset[i] + 1];
        v2->simdInternal_[i] = base[align * offset[i] + 2];
    }
}


/*! \brief Transpose and store 3 SIMD floats to 3 consecutive addresses at
 *         GMX_SIMD_FLOAT_WIDTH offsets.
 *
 * \tparam     align  Alignment of the memory to which we write, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 3 for this routine) the output data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are written.
 * \param[out] base   Pointer to the start of the memory area
 * \param      offset Aligned array with offsets to the start of each triplet.
 * \param      v0     1st component of triplets, written to base[align*offset[i]].
 * \param      v1     2nd component of triplets, written to base[align*offset[i] + 1].
 * \param      v2     3rd component of triplets, written to base[align*offset[i] + 2].
 *
 * This function can work with both aligned (better performance) and unaligned
 * memory. When the align parameter is not a power-of-two (align==3 would be normal
 * for packed atomic coordinates) the memory obviously cannot be aligned, and
 * we account for this.
 * However, in the case where align is a power-of-two, we assume the base pointer
 * also has the same alignment, which will enable many platforms to use faster
 * aligned memory store operations.
 * An easy way to think of this is that each triplet of data in memory must be
 * aligned to the align parameter you specify when it's a power-of-two.
 *
 * The offset memory must always be aligned to GMX_SIMD_FINT32_WIDTH, since this
 * enables us to use SIMD loads and gather operations on platforms that support it.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This routine uses a normal array for the offsets, since we typically
 *       load the data from memory. On the architectures we have tested this
 *       is faster even when a SIMD integer datatype is present.
 */
template<int align>
static inline void gmx_simdcall
transposeScatterStoreU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);

    for (std::size_t i = 0; i < v0.simdInternal_.size(); i++)
    {
        base[align * offset[i]]     = v0.simdInternal_[i];
        base[align * offset[i] + 1] = v1.simdInternal_[i];
        base[align * offset[i] + 2] = v2.simdInternal_[i];
    }
}


/*! \brief Transpose and add 3 SIMD floats to 3 consecutive addresses at
 *         GMX_SIMD_FLOAT_WIDTH offsets.
 *
 * \tparam     align  Alignment of the memory to which we write, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 3 for this routine) the output data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are incremented.
 * \param[out] base   Pointer to the start of the memory area
 * \param      offset Aligned array with offsets to the start of each triplet.
 * \param      v0     1st component of triplets, added to base[align*offset[i]].
 * \param      v1     2nd component of triplets, added to base[align*offset[i] + 1].
 * \param      v2     3rd component of triplets, added to base[align*offset[i] + 2].
 *
 * This function can work with both aligned (better performance) and unaligned
 * memory. When the align parameter is not a power-of-two (align==3 would be normal
 * for packed atomic coordinates) the memory obviously cannot be aligned, and
 * we account for this.
 * However, in the case where align is a power-of-two, we assume the base pointer
 * also has the same alignment, which will enable many platforms to use faster
 * aligned memory load/store operations.
 * An easy way to think of this is that each triplet of data in memory must be
 * aligned to the align parameter you specify when it's a power-of-two.
 *
 * The offset memory must always be aligned to GMX_SIMD_FINT32_WIDTH, since this
 * enables us to use SIMD loads and gather operations on platforms that support it.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This routine uses a normal array for the offsets, since we typically
 *       load the data from memory. On the architectures we have tested this
 *       is faster even when a SIMD integer datatype is present.
 * \note To improve performance, this function might use full-SIMD-width
 *       unaligned load/store, and add 0.0 to the extra elements.
 *       This means you need to ensure the memory is padded
 *       at the end, so we always can load GMX_SIMD_REAL_WIDTH elements
 *       starting at the last offset. If you use the Gromacs aligned memory
 *       allocation routines this will always be the case.
 */
template<int align>
static inline void gmx_simdcall
transposeScatterIncrU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);

    for (std::size_t i = 0; i < v0.simdInternal_.size(); i++)
    {
        base[align * offset[i]] += v0.simdInternal_[i];
        base[align * offset[i] + 1] += v1.simdInternal_[i];
        base[align * offset[i] + 2] += v2.simdInternal_[i];
    }
}


/*! \brief Transpose and subtract 3 SIMD floats to 3 consecutive addresses at
 *         GMX_SIMD_FLOAT_WIDTH offsets.
 *
 * \tparam     align  Alignment of the memory to which we write, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 3 for this routine) the output data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are decremented.
 * \param[out] base    Pointer to start of memory.
 * \param      offset  Aligned array with offsets to the start of each triplet.
 * \param      v0      1st component, subtracted from base[align*offset[i]]
 * \param      v1      2nd component, subtracted from base[align*offset[i]+1]
 * \param      v2      3rd component, subtracted from base[align*offset[i]+2]
 *
 * This function can work with both aligned (better performance) and unaligned
 * memory. When the align parameter is not a power-of-two (align==3 would be normal
 * for packed atomic coordinates) the memory obviously cannot be aligned, and
 * we account for this.
 * However, in the case where align is a power-of-two, we assume the base pointer
 * also has the same alignment, which will enable many platforms to use faster
 * aligned memory load/store operations.
 * An easy way to think of this is that each triplet of data in memory must be
 * aligned to the align parameter you specify when it's a power-of-two.
 *
 * The offset memory must always be aligned to GMX_SIMD_FINT32_WIDTH, since this
 * enables us to use SIMD loads and gather operations on platforms that support it.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This routine uses a normal array for the offsets, since we typically
 *       load the data from memory. On the architectures we have tested this
 *       is faster even when a SIMD integer datatype is present.
 * \note To improve performance, this function might use full-SIMD-width
 *       unaligned load/store, and subtract 0.0 from the extra elements.
 *       This means you need to ensure the memory is padded
 *       at the end, so we always can load GMX_SIMD_REAL_WIDTH elements
 *       starting at the last offset. If you use the Gromacs aligned memory
 *       allocation routines this will always be the case.
 */
template<int align>
static inline void gmx_simdcall
transposeScatterDecrU(float* base, const std::int32_t offset[], SimdFloat v0, SimdFloat v1, SimdFloat v2)
{
    // Offset list must be aligned for SIMD FINT32
    assert(std::size_t(offset) % (GMX_SIMD_FINT32_WIDTH * sizeof(std::int32_t)) == 0);

    for (std::size_t i = 0; i < v0.simdInternal_.size(); i++)
    {
        base[align * offset[i]] -= v0.simdInternal_[i];
        base[align * offset[i] + 1] -= v1.simdInternal_[i];
        base[align * offset[i] + 2] -= v2.simdInternal_[i];
    }
}


/*! \brief Expand each element of float SIMD variable into three identical
 *         consecutive elements in three SIMD outputs.
 *
 * \param      scalar    Floating-point input, e.g. [s0 s1 s2 s3] if width=4.
 * \param[out] triplets0 First output, e.g. [s0 s0 s0 s1] if width=4.
 * \param[out] triplets1 Second output, e.g. [s1 s1 s2 s2] if width=4.
 * \param[out] triplets2 Third output, e.g. [s2 s3 s3 s3] if width=4.
 *
 * This routine is meant to use for things like scalar-vector multiplication,
 * where the vectors are stored in a merged format like [x0 y0 z0 x1 y1 z1 ...],
 * while the scalars are stored as [s0 s1 s2...], and the data cannot easily
 * be changed to SIMD-friendly layout.
 *
 * In this case, load 3 full-width SIMD variables from the vector array (This
 * will always correspond to GMX_SIMD_FLOAT_WIDTH triplets),
 * load a single full-width variable from the scalar array, and
 * call this routine to expand the data. You can then simply multiply the
 * first, second and third pair of SIMD variables, and store the three
 * results back into a suitable vector-format array.
 */
static inline void gmx_simdcall expandScalarsToTriplets(SimdFloat  scalar,
                                                        SimdFloat* triplets0,
                                                        SimdFloat* triplets1,
                                                        SimdFloat* triplets2)
{
    for (std::size_t i = 0; i < scalar.simdInternal_.size(); i++)
    {
        triplets0->simdInternal_[i] = scalar.simdInternal_[i / 3];
        triplets1->simdInternal_[i] = scalar.simdInternal_[(i + scalar.simdInternal_.size()) / 3];
        triplets2->simdInternal_[i] = scalar.simdInternal_[(i + 2 * scalar.simdInternal_.size()) / 3];
    }
}

/*! \brief Load 4 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets
 *         specified by a SIMD integer, transpose into 4 SIMD float variables.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 4 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Aligned pointer to the start of the memory.
 * \param      offset SIMD integer type with offsets to the start of each triplet.
 * \param[out] v0     First component, base[align*offset[i]] for each i.
 * \param[out] v1     Second component, base[align*offset[i] + 1] for each i.
 * \param[out] v2     Third component, base[align*offset[i] + 2] for each i.
 * \param[out] v3     Fourth component, base[align*offset[i] + 3] for each i.
 *
 * The floating-point memory locations must be aligned, but only to the smaller
 * of four elements and the floating-point SIMD width.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This is a special routine primarily intended for loading Gromacs
 *       table data as efficiently as possible - this is the reason for using
 *       a SIMD offset index, since the result of the  real-to-integer conversion
 *       is present in a SIMD register just before calling this routine.
 */
template<int align>
static inline void gmx_simdcall gatherLoadBySimdIntTranspose(const float* base,
                                                             SimdFInt32   offset,
                                                             SimdFloat*   v0,
                                                             SimdFloat*   v1,
                                                             SimdFloat*   v2,
                                                             SimdFloat*   v3)
{
    // Base pointer must be aligned to the smaller of 4 elements and float SIMD width
    assert(std::size_t(base) % (std::min(GMX_SIMD_FLOAT_WIDTH, 4) * sizeof(float)) == 0);
    // align parameter must also be a multiple of the above alignment requirement
    assert(align % std::min(GMX_SIMD_FLOAT_WIDTH, 4) == 0);

    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset.simdInternal_[i]];
        v1->simdInternal_[i] = base[align * offset.simdInternal_[i] + 1];
        v2->simdInternal_[i] = base[align * offset.simdInternal_[i] + 2];
        v3->simdInternal_[i] = base[align * offset.simdInternal_[i] + 3];
    }
}


/*! \brief Load 2 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets
 *         (unaligned) specified by SIMD integer, transpose into 2 SIMD floats.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 2 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Pointer to the start of the memory.
 * \param      offset SIMD integer type with offsets to the start of each triplet.
 * \param[out] v0     First component, base[align*offset[i]] for each i.
 * \param[out] v1     Second component, base[align*offset[i] + 1] for each i.
 *
 * Since some SIMD architectures cannot handle any unaligned loads, this routine
 * is only available if GMX_SIMD_HAVE_GATHER_LOADU_BYSIMDINT_TRANSPOSE is 1.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This is a special routine primarily intended for loading Gromacs
 *       table data as efficiently as possible - this is the reason for using
 *       a SIMD offset index, since the result of the  real-to-integer conversion
 *       is present in a SIMD register just before calling this routine.
 */
template<int align>
static inline void gmx_simdcall
gatherLoadUBySimdIntTranspose(const float* base, SimdFInt32 offset, SimdFloat* v0, SimdFloat* v1)
{
    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset.simdInternal_[i]];
        v1->simdInternal_[i] = base[align * offset.simdInternal_[i] + 1];
    }
}

/*! \brief Load 2 consecutive floats from each of GMX_SIMD_FLOAT_WIDTH offsets
 *         specified by a SIMD integer, transpose into 2 SIMD float variables.
 *
 * \tparam     align  Alignment of the memory from which we read, i.e. distance
 *                    (measured in elements, not bytes) between index points.
 *                    When this is identical to the number of SIMD variables
 *                    (i.e., 2 for this routine) the input data is packed without
 *                    padding in memory. See the SIMD parameters for exactly
 *                    what memory positions are loaded.
 * \param      base   Aligned pointer to the start of the memory.
 * \param      offset SIMD integer type with offsets to the start of each triplet.
 * \param[out] v0     First component, base[align*offset[i]] for each i.
 * \param[out] v1     Second component, base[align*offset[i] + 1] for each i.
 *
 * The floating-point memory locations must be aligned, but only to the smaller
 * of two elements and the floating-point SIMD width.
 *
 * \note You should NOT scale offsets before calling this routine; it is
 *       done internally by using the alignment template parameter instead.
 * \note This is a special routine primarily intended for loading Gromacs
 *       table data as efficiently as possible - this is the reason for using
 *       a SIMD offset index, since the result of the  real-to-integer conversion
 *       is present in a SIMD register just before calling this routine.
 */
template<int align>
static inline void gmx_simdcall
gatherLoadBySimdIntTranspose(const float* base, SimdFInt32 offset, SimdFloat* v0, SimdFloat* v1)
{
    // Base pointer must be aligned to the smaller of 2 elements and float SIMD width
    assert(std::size_t(base) % (std::min(GMX_SIMD_FLOAT_WIDTH, 2) * sizeof(float)) == 0);
    // align parameter must also be a multiple of the above alignment requirement
    assert(align % std::min(GMX_SIMD_FLOAT_WIDTH, 2) == 0);

    for (std::size_t i = 0; i < v0->simdInternal_.size(); i++)
    {
        v0->simdInternal_[i] = base[align * offset.simdInternal_[i]];
        v1->simdInternal_[i] = base[align * offset.simdInternal_[i] + 1];
    }
}


/*! \brief Reduce each of four SIMD floats, add those values to four consecutive
 *         floats in memory, return sum.
 *
 * \param m   Pointer to memory where four floats should be incremented
 * \param v0  SIMD variable whose sum should be added to m[0]
 * \param v1  SIMD variable whose sum should be added to m[1]
 *
 * \return Sum of all elements in the four SIMD variables.
 *
 * The pointer m must be aligned to the smaller of two elements and the
 * floating-point SIMD width.
 *
 * \note This is a special routine intended for the Gromacs nonbonded kernels.
 * It is used in the epilogue of the outer loop, where the variables will
 * contain unrolled forces for one outer-loop-particle each, corresponding to
 * a single coordinate (i.e, say, four x-coordinate force variables). These
 * should be summed and added to the force array in memory. Since we always work
 * with contiguous SIMD-layout , we can use efficient aligned loads/stores.
 * When calculating the virial, we also need the total sum of all forces for
 * each coordinate. This is provided as the return value. For routines that
 * do not need these, this extra code will be optimized away completely if you
 * just ignore the return value (Checked with gcc-4.9.1 and clang-3.6 for AVX).
 */
static inline float gmx_simdcall reduceIncr2ReturnSum(float* m, SimdFloat v0, SimdFloat v1)
{
    float sum[2]; // Note that the 2 here corresponds to the 2 m-elements, not any SIMD width

    // Make sure the memory pointer is aligned to the smaller of 2 elements and float SIMD width
    assert(std::size_t(m) % (std::min(GMX_SIMD_FLOAT_WIDTH, 2) * sizeof(float)) == 0);

    sum[0] = reduce(v0);
    sum[1] = reduce(v1);

    m[0] += sum[0];
    m[1] += sum[1];

    return sum[0] + sum[1];
}


/*! \brief Reduce each of four SIMD floats, add those values to four consecutive
 *         floats in memory, return sum.
 *
 * \param m   Pointer to memory where four floats should be incremented
 * \param v0  SIMD variable whose sum should be added to m[0]
 * \param v1  SIMD variable whose sum should be added to m[1]
 * \param v2  SIMD variable whose sum should be added to m[2]
 * \param v3  SIMD variable whose sum should be added to m[3]
 *
 * \return Sum of all elements in the four SIMD variables.
 *
 * The pointer m must be aligned to the smaller of four elements and the
 * floating-point SIMD width.
 *
 * \note This is a special routine intended for the Gromacs nonbonded kernels.
 * It is used in the epilogue of the outer loop, where the variables will
 * contain unrolled forces for one outer-loop-particle each, corresponding to
 * a single coordinate (i.e, say, four x-coordinate force variables). These
 * should be summed and added to the force array in memory. Since we always work
 * with contiguous SIMD-layout , we can use efficient aligned loads/stores.
 * When calculating the virial, we also need the total sum of all forces for
 * each coordinate. This is provided as the return value. For routines that
 * do not need these, this extra code will be optimized away completely if you
 * just ignore the return value (Checked with gcc-4.9.1 and clang-3.6 for AVX).
 */
static inline float gmx_simdcall reduceIncr4ReturnSum(float* m, SimdFloat v0, SimdFloat v1, SimdFloat v2, SimdFloat v3)
{
    float sum[4]; // Note that the 4 here corresponds to the 4 m-elements, not any SIMD width

    // Make sure the memory pointer is aligned to the smaller of 4 elements and float SIMD width
    assert(std::size_t(m) % (std::min(GMX_SIMD_FLOAT_WIDTH, 4) * sizeof(float)) == 0);

    sum[0] = reduce(v0);
    sum[1] = reduce(v1);
    sum[2] = reduce(v2);
    sum[3] = reduce(v3);

    m[0] += sum[0];
    m[1] += sum[1];
    m[2] += sum[2];
    m[3] += sum[3];

    return sum[0] + sum[1] + sum[2] + sum[3];
}

/*! \} */

/*! \} */
/*! \endcond */

} // namespace gmx

#endif // GMX_SIMD_IMPL_RISCV_V_UTIL_FLOAT_H
