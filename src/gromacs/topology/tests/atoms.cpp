/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2019- The GROMACS Authors
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
/*! \internal \file
 * \brief
 * Tests for atoms datastructures
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/topology/atoms.h"

#include <memory>

#include <gtest/gtest.h>

#include "gromacs/topology/symtab.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/inmemoryserializer.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/refdata.h"

namespace gmx
{

namespace test
{

class SimulationResidueTest : public ::testing::Test
{
public:
    SimulationResidueTest();
    //! Access to table builder.
    StringTableBuilder* builder() { return &tableBuilder_; }

private:
    //! Need a string table with test strings, initialized during setup.
    StringTableBuilder tableBuilder_;
    //! Handler for reference data.
    TestReferenceData data_;
    //! Handler for checking reference data.
    std::unique_ptr<TestReferenceChecker> checker_;
};

SimulationResidueTest::SimulationResidueTest()
{
    std::string nameString = "residue name";
    std::string rtpString  = "rtp name";
    tableBuilder_.addString(nameString);
    tableBuilder_.addString(rtpString);
}

TEST_F(SimulationResidueTest, CanCreateFullWithConstructor)
{
    auto              table = builder()->build();
    SimulationResidue testResidue(table.at(0), 0, ' ', 1, ' ', table.at(1));
    EXPECT_STREQ(testResidue.name().c_str(), "residue name");
    EXPECT_EQ(testResidue.nr(), 0);
    EXPECT_EQ(testResidue.insertionCode(), ' ');
    EXPECT_EQ(testResidue.chainNumber(), 1);
    EXPECT_EQ(testResidue.chainIdentifier(), ' ');
}

TEST_F(SimulationResidueTest, CanWriteToSerializer)
{
    auto                    table = builder()->build();
    SimulationResidue       testResidue(table.at(0), 0, ' ', 1, ' ', table.at(1));
    gmx::InMemorySerializer writer;
    testResidue.serializeResidue(&writer);
    auto buffer = writer.finishAndGetBuffer();
    EXPECT_EQ(buffer.size(), 9); // 4 (index for name) + 4 (residue index) + 1 (insertion code)
}

TEST_F(SimulationResidueTest, RoundTrip)
{
    auto                    table = builder()->build();
    SimulationResidue       testResidue(table.at(0), 0, ' ', 1, ' ', table.at(1));
    gmx::InMemorySerializer writer;
    testResidue.serializeResidue(&writer);
    auto buffer = writer.finishAndGetBuffer();
    EXPECT_EQ(buffer.size(), 9); // 4 (index for name) + 4 (residue index) + 1 (insertion code)

    gmx::InMemoryDeserializer reader(buffer, false);
    SimulationResidue         newResidue(&reader, table);
    EXPECT_STREQ(testResidue.name().c_str(), newResidue.name().c_str());
    EXPECT_EQ(testResidue.nr(), newResidue.nr());
    EXPECT_EQ(testResidue.insertionCode(), newResidue.insertionCode());
}

} // namespace test

} // namespace gmx
