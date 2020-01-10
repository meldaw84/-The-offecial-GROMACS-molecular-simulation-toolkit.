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
#include <string>

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

namespace
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

TEST(SimulationParticleTest, CanCreateWithIdenticalABStates)
{
    std::string        elem("foo\n");
    ParticleMass       mass{ 0.1 };
    ParticleCharge     charge{ 0.2 };
    ParticleTypeName   type{ 1 };
    SimulationParticle testParticle(mass, charge, type, ParticleType::Atom, 3, 4, elem);
    EXPECT_FLOAT_EQ(testParticle.m(), 0.1);
    EXPECT_FLOAT_EQ(testParticle.m(), testParticle.mB());
    EXPECT_FLOAT_EQ(testParticle.q(), 0.2);
    EXPECT_FLOAT_EQ(testParticle.q(), testParticle.qB());
    EXPECT_EQ(testParticle.type(), 1);
    EXPECT_EQ(testParticle.type(), testParticle.typeB());
    EXPECT_EQ(testParticle.ptype(), ParticleType::Atom);
    EXPECT_EQ(testParticle.resind(), 3);
    EXPECT_EQ(testParticle.atomnumber(), 4);
    EXPECT_STREQ(testParticle.elem().c_str(), elem.c_str());
}

TEST(SimulationParticleTest, CanCreateWithDifferentABStates)
{
    std::string        elem("foo\n");
    ParticleMass       mass   = { 0.1, 0.3 };
    ParticleCharge     charge = { 0.2, 0.4 };
    ParticleTypeName   type   = { 1, 4 };
    SimulationParticle testParticle(mass, charge, type, ParticleType::Bond, 3, 4, elem);
    EXPECT_FLOAT_EQ(testParticle.m(), 0.1);
    EXPECT_NE(testParticle.m(), testParticle.mB());
    EXPECT_FLOAT_EQ(testParticle.mB(), 0.3);
    EXPECT_FLOAT_EQ(testParticle.q(), 0.2);
    EXPECT_NE(testParticle.q(), testParticle.qB());
    EXPECT_FLOAT_EQ(testParticle.qB(), 0.4);
    EXPECT_EQ(testParticle.type(), 1);
    EXPECT_NE(testParticle.type(), testParticle.typeB());
    EXPECT_EQ(testParticle.typeB(), 4);
    EXPECT_EQ(testParticle.ptype(), ParticleType::Bond);
    EXPECT_EQ(testParticle.resind(), 3);
    EXPECT_EQ(testParticle.atomnumber(), 4);
    EXPECT_STREQ(testParticle.elem().c_str(), elem.c_str());
}

TEST(SimulationParticleTest, CanWriteToSerializer)
{
    std::string             elem("foo\n");
    ParticleMass            mass   = { 0.1, 0.3 };
    ParticleCharge          charge = { 0.2, 0.4 };
    ParticleTypeName        type   = { 1, 4 };
    SimulationParticle      testParticle(mass, charge, type, ParticleType::Nucleus, 3, 4, elem);
    gmx::InMemorySerializer writer;
    testParticle.serializeParticle(&writer);
    auto buffer = writer.finishAndGetBuffer();
    int  expectedResult =
            GMX_DOUBLE ? 54 : 38; // 2 * (2 * real + bool) + (2 * ushort + bool) + 3 * int + 3 * bool
    EXPECT_EQ(buffer.size(), expectedResult);
}

TEST(SimulationParticleTest, RoundTrip)
{
    std::string             elem("foo\n");
    ParticleMass            mass   = { 0.1, 0.3 };
    ParticleCharge          charge = { 0.2, 0.4 };
    ParticleTypeName        type   = { 1, 4 };
    SimulationParticle      testParticle(mass, charge, type, ParticleType::Shell, 3, 4, elem);
    gmx::InMemorySerializer writer;
    testParticle.serializeParticle(&writer);
    auto buffer = writer.finishAndGetBuffer();
    int  expectedResult =
            GMX_DOUBLE ? 54 : 38; // 2 * (2 * real + bool) + (2 * ushort + bool) + 3 * int + 3 * bool
    EXPECT_EQ(buffer.size(), expectedResult);

    gmx::InMemoryDeserializer reader(buffer, GMX_DOUBLE);
    SimulationParticle        newParticle(&reader);
    EXPECT_FLOAT_EQ(testParticle.m(), newParticle.m());
    EXPECT_FLOAT_EQ(testParticle.mB(), newParticle.mB());
    EXPECT_FLOAT_EQ(testParticle.q(), newParticle.q());
    EXPECT_FLOAT_EQ(testParticle.qB(), newParticle.qB());
    EXPECT_EQ(testParticle.type(), newParticle.type());
    EXPECT_EQ(testParticle.typeB(), newParticle.typeB());
    EXPECT_EQ(testParticle.ptype(), newParticle.ptype());
    EXPECT_EQ(testParticle.resind(), newParticle.resind());
    EXPECT_EQ(testParticle.atomnumber(), newParticle.atomnumber());
}

} // namespace

} // namespace test

} // namespace gmx
