/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022-, by the GROMACS development team, led by
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
/*! \internal \file
 *
 * \brief Tests routines in chargesetter.h .
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 */

#include "gmxpre.h"

#include <gtest/gtest.h>

#include "gromacs/domdec/localatomsetmanager.h"
#include "gromacs/mdlib/chargesettermanager.h"
#include "gromacs/utility/arrayref.h"

namespace gmx
{
namespace test
{
namespace
{

TEST(ChargeSetterManagerTest, canConstruct)
{
    std::array<real, 4> chargeA = { 1, 1, 1, 1 };
    std::array<real, 4> chargeB = { 2, 2, 2, 2 };

    LocalAtomSetManager atomSets;

    std::array<index, 2> atomIndices = { 2, 3 };
    LocalAtomSet         atomSet     = atomSets.add(makeConstArrayRef(atomIndices));

    std::array<index, 1> otherAtomIndices = { 0 };
    LocalAtomSet         atomSetB         = atomSets.add(makeConstArrayRef(otherAtomIndices));


    ChargeSetterManager chargeManager(makeArrayRef(chargeA), makeConstArrayRef(chargeB));

    ChargeSetter chargeSetter      = chargeManager.add(atomSet);
    ChargeSetter otherChargeSetter = chargeManager.add(otherAtomSet);

    chargeSetter.requestSetCharge(-1);

    std::array<real, 4> expectedChargeA = { 1, 1, 1, 1 };
    EXPECT_THAT(chargeA, Pointwise(Eq(), expectedChargeA));

    chargeManager.updateCharges();

    fprintf(stderr, "\nPerformed update  : %g %g %g %g\n", chargeA[0], chargeA[1], chargeA[2], chargeA[3]);

    otherChargeSetter.requestSetCharge(3);

    chargeManager.updateCharges();

    fprintf(stderr, "\nUpdate first atom : %g %g %g %g\n", chargeA[0], chargeA[1], chargeA[2], chargeA[3]);
}

} // namespace
} // namespace test
} // namespace gmx
