/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2021, by the GROMACS development team, led by
 * Mark Abraham, David van der Spoel, Berk Hess, and Erik Lindahl,
 * and including many others, as listed in the AUTHORS file in the
 * top-level source directory and at http://www.gromacs.org.
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
 * http://www.gnu.org/licenses, or write to the Free Software Foundation,
 * Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301  USA.
 *
 * If you want to redistribute modifications to GROMACS, please
 * consider that scientific software is very special. Version
 * control is crucial - bugs must be traceable. We will be happy to
 * consider code for inclusion in the official distribution, but
 * derived work must not be called official GROMACS. Details are found
 * in the README & COPYING files - if they are missing, get the
 * official version at http://www.gromacs.org.
 *
 * To help us fund GROMACS development, we humbly ask that you cite
 * the research papers on the package. Check out http://www.gromacs.org.
 */

/*! \file
 * \brief Tools for managing data references across the API boundary.
 *
 * \author M. Eric Irrgang <ericirrgang@gmail.com>
 * \ingroup gmxapi_compat
 */

#ifndef GMXAPICOMPAT_DATA_H
#define GMXAPICOMPAT_DATA_H

#include <vector>

#include "gmxapi/gmxapicompat.h"

namespace gmxapicompat
{

// Declare an overload set for a utility function. Provide a default implementation.
template<class T>
auto bytesPrecision(T& obj) -> decltype(T::bytesPrecision())
{
    return obj.bytesPrecision();
}

/*!
 * \brief Handle for managing access and lifetime to data resources.
 *
 * DataObjects are always allocated and deallocated in the gmxapi library
 * to avoid potential ABI incompatibilities between library and client code.
 */
class DataHandle
{
public:
    class DataObject;

    explicit DataHandle(DataObject*);
    ~DataHandle();

    DataObject* get();

private:
    DataObject* obj_;
};

/*!
 * \brief Buffer descriptor for GROMACS data.
 *
 * This structure should be sufficient to map a GROMACS managed array to common
 * buffer protocols for API array data exchange.
 *
 * \warning Data may be internally stored as 32-bit floating point numbers, but
 * GROMACS developers have not yet agreed to include 32-bit floating point data in the gmxapi
 * data typing specification.
 */
struct BufferDescription
{
    void*               ptr;
    gmxapi::GmxapiType  itemType;
    size_t              itemSize;
    size_t              ndim;
    std::vector<size_t> shape;
    std::vector<size_t> strides;
    bool                writeable;
};

/*!
 * \brief Get a buffer description for a data handle, if possible.
 *
 * \return buffer description object (does not guarantee continued validity)
 *
 * Caller is responsible for continuing to hold the DataHandle while the BufferDescription is in
 * use.
 *
 * Future API should allow a scoped, stable (reference-counted) interface to be acquired, but
 * discussion is warranted on appropriate access policies, guarantees on data (non-)conversion
 * and layout, and syntax.
 */
BufferDescription buffer_description(const DataHandle&);

/*!
 * \brief Check a struct for consistency with the gmxapicompat data buffer protocol.
 *
 * \tparam T Buffer descriptor type.
 * \tparam Enable SFINAE trait checker.
 */
template<typename T, class Enable = void>
struct is_gmxapi_data_buffer : std::false_type
{
};

template<typename T>
struct is_gmxapi_data_buffer<
        T,
        std::enable_if_t<
                std::is_pointer_v<decltype(
                        T::ptr)> && std::is_same_v<decltype(T::itemType), gmxapi::GmxapiType> && std::is_convertible_v<decltype(T::itemSize), size_t> && std::is_convertible_v<decltype(T::ndim), size_t> && std::is_convertible_v<decltype(T::shape), std::vector<size_t>> && std::is_convertible_v<decltype(T::strides), std::vector<size_t>> && std::is_convertible_v<decltype(T::writeable), bool>>> :
    std::true_type
{
};

static_assert(is_gmxapi_data_buffer<BufferDescription>::value,
              "Interface cannot support buffer protocols.");

/*!
 * \brief Floating point precision mismatch.
 *
 * Operation cannot be performed at the requested precision for the provided input.
 *
 * \ingroup gmxapi_exceptions
 */
class PrecisionError : public gmxapi::BasicException<PrecisionError>
{
public:
    using BasicException<PrecisionError>::BasicException;
};

} // end namespace gmxapicompat

#endif // GMXAPICOMPAT_DATA_H
