/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright (c) 2022- by the GROMACS development team, led by
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
 * \brief Implements routines in chargesettermanager.h .
 *
 * \author Christian Blau <cblau.mail@gmail.com>
 */

#include "gmxpre.h"

#include "gromacs/domdec/localatomset.h" // needed until pimpled
#include "gromacs/mdlib/chargesetter.h"
#include "gromacs/mdlib/chargesettermanager.h"
#include "gromacs/utility/arrayref.h"        // needed until pimpled
#include "gromacs/utility/basedefinitions.h" // needed for index until pimpled
#include "gromacs/utility/foreignparameterupdater.h"

namespace gmx
{

namespace
{

struct ChargeSet
{
    ChargeSet(const LocalAtomSet& atoms, real initialInterpolationParameter = 0) :
        atoms_{ atoms }, interpolationParameter_{ initialInterpolationParameter }
    {
    }
    const LocalAtomSet      atoms_;
    ForeignParameterUpdater interpolationParameter_;
};

} // namespace


class ChargeSetterManager::Impl
{
public:
    Impl(ArrayRef<real> chargeA, ArrayRef<const real> chargeB) :
        charge_{ chargeA }, chargeA_(chargeA.begin(), chargeA.end()), chargeB_{ chargeB }
    {
    }

    std::vector<ChargeSet> chargeInterpolationParameters_;
    const ArrayRef<real> charge_; // view on chargeA, which represents the current charge in other parts of the code
    const std::vector<real> chargeA_; // chargeA is misused as *the* "charge" in other parts of the code, that's why we make a copy
    const ArrayRef<const real> chargeB_; // chargeB is never changed, we can keep a constant view on it.
};


ChargeSetterManager::ChargeSetterManager(ArrayRef<real> chargeA, ArrayRef<const real> chargeB) :
    impl_{ std::make_unique<ChargeSetterManager::Impl>(chargeA, chargeB) }
{
}

ChargeSetter ChargeSetterManager::add(const LocalAtomSet& atoms)
{
    // TODO here we must add a check that no two atomsets overlap
    impl_->chargeInterpolationParameters_.emplace_back(atoms, 0);
    return ChargeSetter(
            impl_->chargeInterpolationParameters_.back().interpolationParameter_.handleToRequestParameter());
}

void ChargeSetterManager::updateCharges()
{
    for (ChargeSet& currentChargeSet : impl_->chargeInterpolationParameters_)
    {
        if (currentChargeSet.interpolationParameter_.needsUpdate())
        {
            real interpolationParameter =
                    currentChargeSet.interpolationParameter_.updateAndGetParameter();
            for (const index currentGlobalIndex : currentChargeSet.atoms_.globalIndex())
            {
                impl_->charge_[currentGlobalIndex] =
                        (1 - interpolationParameter) * impl_->chargeA_[currentGlobalIndex]
                        + (interpolationParameter)*impl_->chargeB_[currentGlobalIndex];
            }
        }
    }
}


} // namespace gmx