/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2020- The GROMACS Authors
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
/*! \libinternal \file
 * \brief
 * Declares utilities for template metaprogramming
 *
 * \author Roland Schulz <roland.schulz@intel.com>
 *
 * \inlibraryapi
 * \ingroup module_utility
 */
#ifndef GMX_UTILITY_TEMPLATE_MP_H
#define GMX_UTILITY_TEMPLATE_MP_H

#include <cassert>
#include <cstddef>

#include <utility>

#include "gromacs/compat/mp11.h"

namespace gmx
{

/*! \internal \brief
 * Helper function to select appropriate template based on runtime values.
 *
 * Can use enums, booleans, or ints for template parameters.
 *
 * When using an enum, it must have a member \c Count indicating the
 * total number of valid values.
 *
 * When using an int, the maximum value must be at most 64. Because
 * the underlying implementation is recursive, it is preferable to
 * only use small integer values. If the function being called does
 * not support the full range to 64, it is important to use e.g.  a
 * static_assert to provide a useful message at compilation time.
 *
 * Example usage:
 * \code
    enum class Options {
        Op1 = 0,
        Op2 = 1,
        Count = 2
    };

    template<bool b, Options o1, Options o2>
    bool foo(float f);

    // Note i must be at most 64
    bool bar(bool b, Options o1, Options o2, int i, float f) {
        return dispatchTemplatedFunction(
            [=](auto p0, auto p1, auto p2, auto p3) {
                return foo<p0, p1, p2, p3>(f);
            },
            b, o1, o2, i);
    }
 * \endcode
 *
 * \tparam Function Type of \p f.
 * \param f Function to call.
 * \return The result of calling \c f().
*/
template<class Function>
auto dispatchTemplatedFunction(Function&& f)
{
    return std::forward<Function>(f)();
}

// Recursive templates confuse Doxygen
//! \cond
template<class Function, class Enum, class... Args>
auto dispatchTemplatedFunction(Function&& f, Enum e, Args... args)
{
    return dispatchTemplatedFunction(
            [&](auto... args_) {
                return compat::mp_with_index<size_t(Enum::Count)>(size_t(e), [&](auto e_) {
                    return std::forward<Function>(f)(
                            std::integral_constant<Enum, static_cast<Enum>(size_t(e_))>(), args_...);
                });
            },
            args...);
}

template<class Function, class... Args>
auto dispatchTemplatedFunction(Function&& f, bool b, Args... args)
{
    return dispatchTemplatedFunction(
            [&](auto... args_) {
                return compat::mp_with_index<2>(size_t(b), [&](auto b_) {
                    return std::forward<Function>(f)(std::bool_constant<static_cast<bool>(b_)>(), args_...);
                });
            },
            args...);
}

template<class Function, class... Args>
auto dispatchTemplatedFunction(Function&& f, int i, Args... args)
{
    return dispatchTemplatedFunction(
            [&](auto... args_) {
                return compat::mp_with_index<64>(size_t(i), [&](auto i_) {
                    return std::forward<Function>(f)(
                            std::integral_constant<int, static_cast<int>(i_)>(), args_...);
                });
            },
            args...);
}
//! \endcond

} // namespace gmx

#endif // GMX_UTILITY_TEMPLATE_MP_H
