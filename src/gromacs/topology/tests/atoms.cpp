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

#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "gromacs/topology/symtab.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/inmemoryserializer.h"
#include "gromacs/utility/iserializer.h"
#include "gromacs/utility/strconvert.h"
#include "gromacs/utility/stringutil.h"

#include "testutils/refdata.h"
#include "testutils/testasserts.h"
#include "testutils/testmatchers.h"

#ifndef DOXYGEN

template struct FEPStateValue<real>;
template struct FEPStateValue<unsigned short>;
template struct FEPStateValue<NameHolder>;

namespace gmx
{

namespace test
{

namespace
{

TEST(PdbAtomEntryTest, CanCreateBasicEntry)
{
    PdbAtomEntry testEntry(PdbRecordType::Atom, 1, ' ', "CA");
    EXPECT_EQ(testEntry.type(), PdbRecordType::Atom);
    EXPECT_EQ(testEntry.atomSerialNumber(), 1);
    EXPECT_EQ(testEntry.altloc(), ' ');
    EXPECT_STREQ(testEntry.atomName().c_str(), "CA");
    EXPECT_FALSE(testEntry.occupancy().has_value());
    EXPECT_FALSE(testEntry.bFactor().has_value());
    EXPECT_FALSE(testEntry.anisotropy().has_value());
}

TEST(PdbAtomEntryTest, CanCreateEntryWithOccupAndBfac)
{
    PdbAtomEntry testEntry(PdbRecordType::Atom, 1, ' ', "CA", 0.53, 42.1);
    EXPECT_EQ(testEntry.type(), PdbRecordType::Atom);
    EXPECT_EQ(testEntry.atomSerialNumber(), 1);
    EXPECT_EQ(testEntry.altloc(), ' ');
    EXPECT_STREQ(testEntry.atomName().c_str(), "CA");
    ASSERT_TRUE(testEntry.occupancy().has_value());
    EXPECT_FLOAT_EQ(*testEntry.occupancy(), 0.53);
    ASSERT_TRUE(testEntry.bFactor().has_value());
    EXPECT_FLOAT_EQ(*testEntry.bFactor(), 42.1);
    EXPECT_FALSE(testEntry.anisotropy().has_value());
}

TEST(PdbAtomEntryTest, CanCreateFullEntry)
{
    std::array<real, 6> uij = { 1, 2, 3, 4, 5, 6 };
    PdbAtomEntry        testEntry(PdbRecordType::Atom, 1, ' ', "CA", 0.53, 42.1, uij);
    EXPECT_EQ(testEntry.type(), PdbRecordType::Atom);
    EXPECT_EQ(testEntry.atomSerialNumber(), 1);
    EXPECT_EQ(testEntry.altloc(), ' ');
    EXPECT_STREQ(testEntry.atomName().c_str(), "CA");
    ASSERT_TRUE(testEntry.occupancy().has_value());
    EXPECT_FLOAT_EQ(*testEntry.occupancy(), 0.53);
    ASSERT_TRUE(testEntry.bFactor().has_value());
    EXPECT_FLOAT_EQ(*testEntry.bFactor(), 42.1);
    ASSERT_TRUE(testEntry.anisotropy().has_value());
    auto accessedMatrix = *testEntry.anisotropy();
    EXPECT_THAT(accessedMatrix, ::testing::Pointwise(::testing::Eq(), uij));
}

#endif

template<typename T>
void checkParticleValue(TestReferenceChecker* checker,
                        bool                  haveBState,
                        const T&              valueA,
                        const T&              valueB,
                        const std::string&    fieldName)
{
    TestReferenceChecker compound(checker->checkCompound(fieldName.c_str(), fieldName));
    compound.checkBoolean(haveBState, "HaveBState");
    compound.checkValue(valueA, "ValueA");
    compound.checkValue(valueB, "ValueB");
    EXPECT_EQ(haveBState, valueA != valueB);
}

void checkParticleMiscInfo(TestReferenceChecker* checker,
                           ParticleType          type,
                           gmx::index            resind,
                           int                   atomnumber,
                           const std::string&    elem)
{
    TestReferenceChecker compound(checker->checkCompound("Misc", "Misc"));
    compound.checkInteger(static_cast<int>(type), "TypeAsInt");
    compound.checkInt64(resind, "ResidueIndex");
    compound.checkInteger(atomnumber, "AtomNumber");
    compound.checkString(elem, "ElementString");
}

void checkParticle(TestReferenceChecker* checker, const SimulationParticle& particle)
{
    TestReferenceChecker compound(checker->checkCompound("Particle", nullptr));
    compound.checkBoolean(particle.haveMass(), "HaveMass");
    compound.checkBoolean(particle.haveCharge(), "HaveCharge");
    compound.checkBoolean(particle.haveType(), "HaveType");
    compound.checkBoolean(particle.haveParticleTypeName(), "HaveTypeName");
    compound.checkBoolean(particle.haveParticleName(), "HaveName");
    if (particle.haveParticleName())
    {
        compound.checkString(particle.particleName(), "Name");
    }
    if (particle.haveMass())
    {
        checkParticleValue<real>(
                &compound, particle.haveBStateForAll(), particle.m(), particle.mB(), "Mass");
    }
    if (particle.haveCharge())
    {
        checkParticleValue<real>(
                &compound, particle.haveBStateForAll(), particle.q(), particle.qB(), "Charge");
    }
    if (particle.haveType())
    {
        checkParticleValue<unsigned short>(&compound,
                                           particle.haveBStateForAll(),
                                           particle.type(),
                                           particle.typeB(),
                                           "TypeValue");
    }
    if (particle.haveParticleTypeName())
    {
        checkParticleValue<std::string>(&compound,
                                        particle.haveBStateForAll(),
                                        particle.particleTypeNameA(),
                                        particle.particleTypeNameB(),
                                        "TypeName");
    }
    checkParticleMiscInfo(
            &compound, particle.ptype(), particle.resind(), particle.atomnumber(), particle.elem());
}

void checkParticles(TestReferenceChecker* checker, gmx::ArrayRef<const SimulationParticle> particles)
{
    TestReferenceChecker compound(checker->checkCompound("Particles", "Collection"));
    compound.checkSequence(particles.begin(), particles.end(), "SimulationParticleCollection", checkParticle);
}

using SimulationParticleTestParameters =
        std::tuple<std::optional<ParticleMass>, std::optional<ParticleCharge>, std::optional<ParticleTypeValue>, bool>;

const ParticleMass      testParticleMassOneState{ 0.1 };
const ParticleMass      testParticleMassTwoState{ 0.2, 0.3 };
const ParticleCharge    testParticleChargeOneState{ 0.4 };
const ParticleCharge    testParticleChargeTwoState{ 0.5, 0.6 };
const ParticleTypeValue testParticleTypeValueOneState{ 7 };
const ParticleTypeValue testParticleTypeValueTwoState{ 8, 9 };


class SimulationParticleTest :
    public ::testing::Test,
    public ::testing::WithParamInterface<SimulationParticleTestParameters>
{
public:
    SimulationParticleTest();
    //! Access to table builder.
    StringTableBuilder* builder() { return &tableBuilder_; }
    //! Run tests.
    void runTest(const SimulationParticle& particle);

private:
    //! Need a string table with test strings, initialized during setup.
    StringTableBuilder tableBuilder_;
    //! Handler for reference data.
    TestReferenceData data_;
    //! Handler for checking reference data.
    TestReferenceChecker checker_;
};

SimulationParticleTest::SimulationParticleTest() : checker_(data_.rootChecker())
{
    std::string particleName = "Calpha";
    std::string typeAName    = "cool";
    std::string typeBName    = "boring";
    tableBuilder_.addString(particleName);
    tableBuilder_.addString(typeAName);
    tableBuilder_.addString(typeBName);
}

void SimulationParticleTest::runTest(const SimulationParticle& particle)
{
    checkParticle(&checker_, particle);
}

TEST_P(SimulationParticleTest, CanCreate)
{
    const auto params          = GetParam();
    const auto mass            = std::get<0>(params);
    const auto charge          = std::get<1>(params);
    const auto typeValue       = std::get<2>(params);
    const auto useTwoStateName = std::get<3>(params);
    const auto table           = builder()->build();

    ParticleTypeName typeName = useTwoStateName ? ParticleTypeName(table.at(1), table.at(2))
                                                : ParticleTypeName(table.at(1));

    std::string elem("foo\n");
    runTest(SimulationParticle(
            mass, charge, typeValue, typeName, table.at(0), ParticleType::Atom, 3, 4, elem));
}

TEST_P(SimulationParticleTest, CanSerialize)
{
    const auto params          = GetParam();
    const auto mass            = std::get<0>(params);
    const auto charge          = std::get<1>(params);
    const auto typeValue       = std::get<2>(params);
    const auto useTwoStateName = std::get<3>(params);

    const auto         table    = builder()->build();
    ParticleTypeName   typeName = useTwoStateName ? ParticleTypeName(table.at(1), table.at(2))
                                                  : ParticleTypeName(table.at(1));
    std::string        elem("foo\n");
    SimulationParticle testParticle(
            mass, charge, typeValue, typeName, table.at(0), ParticleType::Atom, 3, 4, elem);

    gmx::InMemorySerializer writer;
    testParticle.serializeParticle(&writer);
    auto                      buffer = writer.finishAndGetBuffer();
    gmx::InMemoryDeserializer reader(buffer, GMX_DOUBLE);
    runTest(SimulationParticle(&reader, table));
}

INSTANTIATE_TEST_SUITE_P(BuildsValidDataStructure,
                         SimulationParticleTest,
                         ::testing::Combine(::testing::Values(std::nullopt,
                                                              std::optional(testParticleMassOneState),
                                                              std::optional(testParticleMassTwoState)),
                                            ::testing::Values(std::nullopt,
                                                              std::optional(testParticleChargeOneState),
                                                              std::optional(testParticleChargeTwoState)),
                                            ::testing::Values(std::nullopt,
                                                              std::optional(testParticleTypeValueOneState),
                                                              std::optional(testParticleTypeValueTwoState)),
                                            ::testing::Values(false, true)));

#if !defined(NDEBUG)
TEST_F(SimulationParticleTest, DeathTestSerialize)
{
    const auto         mass      = testParticleMassOneState;
    const auto         charge    = testParticleChargeOneState;
    const auto         typeValue = testParticleTypeValueOneState;
    SimulationParticle testParticle(
            mass, charge, typeValue, {}, {}, ParticleType::Atom, 3, 4, "foo");
    gmx::InMemorySerializer writer;
    GMX_EXPECT_DEATH_IF_SUPPORTED(testParticle.serializeParticle(&writer),
                                  "Can not access uninitialized element");
}

#else

TEST(DISABLED_SimulationParticleTest, DeathTestSerialize)
{
    ADD_FAILURE() << "Need to have assertions enabled to run tests for program abort";
}

#endif

void checkResidueBuilderValues(TestReferenceChecker*                   checker,
                               const std::string&                      name,
                               unsigned char                           insertionCode,
                               gmx::index                              chainNumber,
                               char                                    chainId,
                               const std::string&                      rtp,
                               gmx::ArrayRef<const SimulationParticle> particles)
{
    TestReferenceChecker compound(checker->checkCompound("ResidueBuilder", "ResidueBuilderValues"));
    compound.checkString(name, "Name");
    compound.checkInteger(insertionCode, "InsertionCode");
    compound.checkInt64(chainNumber, "ChainNumber");
    compound.checkInteger(chainId, "ChainIdentifier");
    compound.checkString(rtp, "RTP");
    compound.checkInteger(particles.size(), "NumberOfParticles");
    checkParticles(&compound, particles);
}

void checkResidueValues(TestReferenceChecker* checker,
                        const std::string&    name,
                        gmx::index            nr,
                        unsigned char         insertionCode,
                        gmx::index            begin,
                        gmx::index            size)
{
    TestReferenceChecker compound(checker->checkCompound("ResidueValues", "ResidueValues"));
    compound.checkString(name, "Name");
    compound.checkInt64(nr, "Number");
    compound.checkInteger(insertionCode, "InsertionCode");
    compound.checkInt64(begin, "Begin");
    compound.checkInt64(size, "Size");
}

void checkResidue(TestReferenceChecker* checker, const SimulationResidue& residue)
{
    TestReferenceChecker compound(checker->checkCompound("Residue", nullptr));
    checkResidueValues(
            &compound, residue.name(), residue.nr(), residue.insertionCode(), residue.begin(), residue.size());
}

void checkResidues(TestReferenceChecker* checker, gmx::ArrayRef<const SimulationResidue> residues)
{
    TestReferenceChecker compound(checker->checkCompound("Residues", "Collection"));
    compound.checkSequence(residues.begin(), residues.end(), "SimulationResidueCollection", checkResidue);
}

class SimulationResidueBuilderTest : public ::testing::Test
{
public:
    SimulationResidueBuilderTest();
    //! Run the test with a single residue still in the builder.
    void runTest(const SimulationResidueBuilder& residue);
    //! Run the test with a single residue after finalizing.
    void runTest(const SimulationResidue& residue);
    //! Access to particles information for test.'
    std::vector<SimulationParticle> particles() { return particles_; }
    //! Access to table.
    const StringTable& table() const { return *table_; }

private:
    //! Need a string table with test strings, initialized during setup.
    StringTableBuilder tableBuilder_;
    //! Table data.
    std::unique_ptr<StringTable> table_;
    //! Data for particles used in the residue.
    std::vector<SimulationParticle> particles_;
    //! Handler for reference data.
    TestReferenceData data_;
    //! Handler for checking reference data.
    TestReferenceChecker checker_;
};

SimulationResidueBuilderTest::SimulationResidueBuilderTest() : checker_(data_.rootChecker())
{
    std::string nameString   = "residue name";
    std::string rtpString    = "rtp name";
    std::string particleName = "Calpha";
    std::string typeAName    = "cool";
    std::string typeBName    = "boring";
    tableBuilder_.addString(nameString);
    tableBuilder_.addString(rtpString);
    tableBuilder_.addString(particleName);
    tableBuilder_.addString(typeAName);
    tableBuilder_.addString(typeBName);

    table_ = std::make_unique<StringTable>(tableBuilder_.build());

    ParticleTypeName typeNameOne{ table_->at(3) };
    ParticleTypeName typeNameTwo{ table_->at(3), table_->at(4) };
    particles_.emplace_back(SimulationParticle(testParticleMassOneState,
                                               testParticleChargeOneState,
                                               testParticleTypeValueOneState,
                                               typeNameOne,
                                               table_->at(2),
                                               ParticleType::Atom,
                                               3,
                                               4,
                                               "Ref\n"));
    particles_.emplace_back(SimulationParticle(testParticleMassTwoState,
                                               testParticleChargeTwoState,
                                               testParticleTypeValueTwoState,
                                               typeNameTwo,
                                               table_->at(2),
                                               ParticleType::Atom,
                                               3,
                                               4,
                                               "Ref\n"));
}

void SimulationResidueBuilderTest::runTest(const SimulationResidueBuilder& residue)
{
    checkResidueBuilderValues(&checker_,
                              residue.name(),
                              residue.insertionCode(),
                              residue.chainNumber(),
                              residue.chainIdentifier(),
                              residue.rtp(),
                              residue.particles());
}

void SimulationResidueBuilderTest::runTest(const SimulationResidue& residue)
{
    checkResidue(&checker_, residue);
}

TEST_F(SimulationResidueBuilderTest, CanCreateFullWithConstructor)
{
    runTest(SimulationResidueBuilder(table().at(0), ' ', 0, ' ', table().at(1), particles()));
}

TEST_F(SimulationResidueBuilderTest, CanUpdateAtoms)
{
    auto localParticles = particles();
    auto residueBuilder =
            SimulationResidueBuilder(table().at(0), ' ', 0, ' ', table().at(1), particles());
    ParticleTypeName typeNameOne{ table().at(3) };
    localParticles.emplace_back(SimulationParticle(testParticleMassOneState,
                                                   testParticleChargeOneState,
                                                   testParticleTypeValueOneState,
                                                   typeNameOne,
                                                   table().at(2),
                                                   ParticleType::Atom,
                                                   3,
                                                   4,
                                                   "Ref\n"));
    residueBuilder.updateParticles(std::move(localParticles));
    runTest(residueBuilder);
}

TEST_F(SimulationResidueBuilderTest, CanCreateResidueFromBuilder)
{
    auto residueBuilder =
            SimulationResidueBuilder(table().at(0), ' ', 0, ' ', table().at(1), particles());
    runTest(residueBuilder);
    auto residue = residueBuilder.finalize(0, 0, residueBuilder.particles().size());
    runTest(residue);

    EXPECT_EQ(residueBuilder.name(), residue.name());
    EXPECT_EQ(residueBuilder.insertionCode(), residue.insertionCode());
}

TEST_F(SimulationResidueBuilderTest, ResidueSerializerWorks)
{
    auto simParticles = particles();
    auto residueBuilder =
            SimulationResidueBuilder(table().at(0), ' ', 0, ' ', table().at(1), particles());
    auto                    residue = residueBuilder.finalize(0, 0, simParticles.size());
    gmx::InMemorySerializer writer;
    residue.serializeResidue(&writer);

    auto                      buffer = writer.finishAndGetBuffer();
    gmx::InMemoryDeserializer reader(buffer, GMX_DOUBLE);

    SimulationResidue readResidue(&reader, table());
    runTest(readResidue);
}

/*! \brief
 * Parameter pack for molecule generation.
 *
 * First tuple: Use BState, use BState for all residues, num residues, num particle in residue
 * Second tuple: Select if BState for mass, charge, type, typeName
 */
using MoleculeBuilderTestParams =
        std::tuple<std::tuple<bool, bool, int, int>, std::tuple<bool, bool, bool, bool>>;

class MoleculeBuilderTest : public ::testing::Test, public ::testing::WithParamInterface<MoleculeBuilderTestParams>
{
public:
    MoleculeBuilderTest();
    //! Add residues to build up the molecule.
    void addResidues();
    //! Finalize the builder to generate the ready object.
    SimulationMolecule finalize();
    //! Access to table.
    const StringTable& table() const { return *table_; }
    //! Access to checker.
    TestReferenceChecker* checker() { return &checker_; }

private:
    //! Stored information.
    SimulationMoleculeBuilder molecule_;
    //! Need a string table with test strings, initialized during setup.
    StringTableBuilder tableBuilder_;
    //! Table data.
    std::unique_ptr<StringTable> table_;
    //! Data for particles used in the residue.
    std::vector<SimulationParticle> particles_;
    //! Handler for reference data.
    TestReferenceData data_;
    //! Handler for checking reference data.
    TestReferenceChecker checker_;
};

MoleculeBuilderTest::MoleculeBuilderTest() : checker_(data_.rootChecker())
{
    std::string nameString   = "residue name";
    std::string rtpString    = "rtp name";
    std::string particleName = "Calpha";
    std::string typeAName    = "cool";
    std::string typeBName    = "boring";
    tableBuilder_.addString(nameString);
    tableBuilder_.addString(rtpString);
    tableBuilder_.addString(particleName);
    tableBuilder_.addString(typeAName);
    tableBuilder_.addString(typeBName);
    table_ = std::make_unique<StringTable>(tableBuilder_.build());
}
void MoleculeBuilderTest::addResidues()
{
    auto params         = GetParam();
    auto residueParams  = std::get<0>(params);
    auto particleParams = std::get<1>(params);
    int  numResidues    = std::get<2>(residueParams);
    int  numParticles   = std::get<3>(residueParams);

    for (int numRes = 0; numRes < numResidues; ++numRes)
    {
        auto useBState = std::get<0>(residueParams) && (std::get<1>(residueParams) || numRes % 2 == 0);
        std::vector<SimulationParticle> particles;
        for (int numPar = 0; numPar < numParticles; ++numPar)
        {
            bool massBState     = useBState && (std::get<0>(particleParams) || numPar % 2 == 0);
            bool chargeBState   = useBState && (std::get<1>(particleParams) || numPar % 2 == 0);
            bool typeBState     = useBState && (std::get<2>(particleParams) || numPar % 2 == 0);
            bool typeNameBState = useBState && (std::get<3>(particleParams) || numPar % 2 == 0);

            auto mass   = massBState ? testParticleMassTwoState : testParticleMassOneState;
            auto charge = chargeBState ? testParticleChargeTwoState : testParticleChargeOneState;
            auto type = typeBState ? testParticleTypeValueTwoState : testParticleTypeValueOneState;
            auto typeName = typeNameBState
                                    ? std::optional(ParticleTypeName(table().at(3), table().at(4)))
                                    : std::optional(ParticleTypeName(table().at(3)));
            auto name     = table().at(2);

            particles.emplace_back(
                    mass, charge, type, typeName, name, ParticleType::Atom, 0, 12, "C");
        }
        molecule_.addResidue(SimulationResidueBuilder(table().at(0), ' ', 0, ' ', table().at(1), &particles));
    }
}

SimulationMolecule MoleculeBuilderTest::finalize()
{
    return molecule_.finalize();
}

void checkMoleculeSanity(TestReferenceChecker* checker, const SimulationMolecule& molecule)
{
    TestReferenceChecker compound(checker->checkCompound("MoleculeData", "Molecule"));
    compound.checkBoolean(molecule.allParticlesHaveChargeAndNotEmpty(), "ChargePresentEverywhere");
    compound.checkBoolean(molecule.allParticlesHaveMassAndNotEmpy(), "MassPresentEverywhere");
    compound.checkBoolean(molecule.allParticlesHaveTypeAndNotEmpty(), "TypePresentEverywhere");
    compound.checkBoolean(molecule.allParticlesHaveBstateAndNotEmpty(), "BStatePresentEverywhere");
    compound.checkBoolean(molecule.allParticlesHaveAtomNameAndNotEmpty(),
                          "AtomNamePresentEverywhere");
    compound.checkBoolean(molecule.allParticlesHaveTypeNameAndNotEmpty(),
                          "AtomTypeNamePresentEverywhere");
    compound.checkInteger(molecule.numResidues(), "NumberOfResidues");
    TestReferenceChecker residueCompound(
            compound.checkCompound("MoleculeResidues", "ResidueValues"));
    checkResidues(&residueCompound, molecule.residues());

    compound.checkInteger(molecule.numParticles(), "NumberOfParticles");
    TestReferenceChecker particleCompound(
            compound.checkCompound("MoleculeParticles", "ParticleValues"));
    checkParticles(&particleCompound, molecule.particles());
}

TEST_F(MoleculeBuilderTest, EmptyStructureGivesFalseForAll)
{
    checkMoleculeSanity(checker(), finalize());
}

TEST_P(MoleculeBuilderTest, ParticlesCheckValidEntries)
{
    addResidues();
    checkMoleculeSanity(checker(), finalize());
}

TEST_P(MoleculeBuilderTest, BuilderIsEmptyAfterFinalize)
{
    addResidues();
    checkMoleculeSanity(checker(), finalize());
    checkMoleculeSanity(checker(), finalize());
}

TEST_P(MoleculeBuilderTest, SerializerWorks)
{
    addResidues();
    gmx::InMemorySerializer writer;
    auto                    molecule = finalize();
    molecule.serializeMolecule(&writer);

    auto                      buffer = writer.finishAndGetBuffer();
    gmx::InMemoryDeserializer reader(buffer, GMX_DOUBLE);

    SimulationMolecule readMolecule(&reader, table());
    checkMoleculeSanity(checker(), readMolecule);
}

INSTANTIATE_TEST_SUITE_P(DetectsValidEntries,
                         MoleculeBuilderTest,
                         ::testing::Combine(::testing::Combine(::testing::Values(true, false),
                                                               ::testing::Values(true, false),
                                                               ::testing::Values(1, 2, 3),
                                                               ::testing::Values(2, 3)),
                                            ::testing::Combine(::testing::Values(true, false),
                                                               ::testing::Values(true, false),
                                                               ::testing::Values(true, false),
                                                               ::testing::Values(true, false))));
} // namespace

} // namespace test

} // namespace gmx
