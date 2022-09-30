/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2022- The GROMACS Authors
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

/*
 * VkFFT hipSYCL support to GROMACS was contributed by Advanced Micro Devices, Inc.
 * Copyright (c) 2022, Advanced Micro Devices, Inc.  All rights reserved.
 */

/*! \internal \file
 *  \brief Defines optimized float3 type for AMD CDNA2 architecture
 *
 *  \author Anton Gorenko <anton@streamhpc.com>
 *  \author Bálint Soproni <balint@streamhpc.com>
 */

#ifndef GMX_GPU_UTILS_VECTYPE_OPS_SYCL_H
#define GMX_GPU_UTILS_VECTYPE_OPS_SYCL_H


/*! \brief Special implementation of float3 for faster computations using packed math on gfx90a.
 *
 * HIP's float3 is defined as a struct of 3 fields, the compiler is not aware of its vector nature
 * hence it is not able to generate optimal packed math instructions (v_pk_...). This new type is
 * defined as struct of float2 (x, y) and float (z) so packed math can be used for x and y.
 */
struct AmdCdna2PackedFloat3
{
    typedef float __attribute__((ext_vector_type(2))) Native_float2_;

    union
    {
        struct __attribute__((packed))
        {
            Native_float2_ dxy;
            float          dz;
        };
        struct
        {
            float x, y, z;
        };
        float array[3];
    };


    AmdCdna2PackedFloat3() = default;


    AmdCdna2PackedFloat3(float x_, float y_, float z_) : dxy{ x_, y_ }, dz{ z_ } {}


    AmdCdna2PackedFloat3(Native_float2_ xy_, float z_) : dxy{ xy_ }, dz{ z_ } {}


    operator float3() const { return float3{ x, y, z }; }


    AmdCdna2PackedFloat3& operator=(const AmdCdna2PackedFloat3& x)
    {
        dxy = x.dxy;
        dz  = x.dz;
        return *this;
    }

    float& operator[](int idx) { return this->array[idx]; }

    const float& operator[](int idx) const { return this->array[idx]; }
};
static_assert(sizeof(AmdCdna2PackedFloat3) == 12);

__forceinline__ AmdCdna2PackedFloat3 operator*(AmdCdna2PackedFloat3 x, AmdCdna2PackedFloat3 y)
{
    return AmdCdna2PackedFloat3{ x.dxy * y.dxy, x.dz * y.dz };
}

__forceinline__ AmdCdna2PackedFloat3 operator*(AmdCdna2PackedFloat3 x, float y)
{
    return AmdCdna2PackedFloat3{ x.dxy * y, x.dz * y };
}

__forceinline__ AmdCdna2PackedFloat3 operator*(float x, AmdCdna2PackedFloat3 y)
{
    return AmdCdna2PackedFloat3{ x * y.dxy, x * y.dz };
}

__forceinline__ AmdCdna2PackedFloat3 operator+(AmdCdna2PackedFloat3 x, AmdCdna2PackedFloat3 y)
{
    return AmdCdna2PackedFloat3{ x.dxy + y.dxy, x.dz + y.dz };
}

__forceinline__ AmdCdna2PackedFloat3 operator-(AmdCdna2PackedFloat3 x, AmdCdna2PackedFloat3 y)
{
    return AmdCdna2PackedFloat3{ x.dxy - y.dxy, x.dz - y.dz };
}

static __forceinline__ AmdCdna2PackedFloat3 make_AmdCdna2PackedFloat3(float x)
{
    return AmdCdna2PackedFloat3{ x, x, x };
}

static __forceinline__ AmdCdna2PackedFloat3 make_AmdCdna2PackedFloat3(float4 x)
{
    return AmdCdna2PackedFloat3{ x.x, x.y, x.z };
}

static __forceinline__ float norm2(AmdCdna2PackedFloat3 a)
{
    AmdCdna2PackedFloat3 b = a * a;
    return (b.x + b.y + b.z);
}

#endif /* GMX_GPU_UTILS_VECTYPE_OPS_SYCL_H */