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
/*! \internal \file
 * \brief
 * Tests for functionality of the "dssp" trajectory analysis module.
 *
 * \author Sergey Gorelov <gorelov_sv@pnpi.nrcki.ru>
 * \author Anatoly Titov <titov_ai@pnpi.nrcki.ru>
 * \author Alexey Shvetsov <alexxyum@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "gromacs/trajectoryanalysis/modules/dssp.h"

#include <regex>
#include <string>

#include <gtest/gtest-param-test.h>
#include <gtest/gtest.h>

#include "testutils/cmdlinetest.h"
#include "testutils/textblockmatchers.h"
#include "gromacs/utility/stringutil.h"

#include "moduletest.h"

namespace gmx
{
namespace test
{
namespace
{

/********************************************************************
 * Tests for gmx::analysismodules::Dssp.
 */

using DsspTestParamsDSSPNB = std::tuple<const char*, real>;
using DsspTestParamsGROMACSNB = std::tuple<const char*, real>;
using DsspTestParamsDSSPNoNB = const char*;
using DsspTestParamsGROMACSNoNB = const char*;

//! Test fixture for the dssp analysis module.
class DsspModuleTestDSSPNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsDSSPNB>
{
};

//! Test fixture for the dssp analysis module.
class DsspModuleTestGROMACSNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsGROMACSNB>
{
};
//! Test fixture for the dssp analysis module.
class DsspModuleTestDSSPNoNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsDSSPNoNB>
{
};

//! Test fixture for the dssp analysis module.
class DsspModuleTestGROMACSNoNB :
    public TrajectoryAnalysisModuleTestFixture<gmx::analysismodules::DsspInfo>,
    public ::testing::WithParamInterface<DsspTestParamsGROMACSNoNB>
{
};

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestDSSPNB, )
{
    auto              params    = GetParam();
    const char* const cmdline[] = { "dssp" };
    std::string       fin(std::get<0>(params));
    // replace pdb in filename with dat to construct uniq output names
    std::string finname = std::regex_replace(fin, std::regex("\\.pdb"), "");
    CommandLine command(cmdline);
    setTopology(fin.c_str());
    setTrajectory(fin.c_str());
    setOutputFile("-o", formatString("%s-dssp-nb-%.1f.dat", finname.c_str(), std::get<1>(params)).c_str(), ExactTextMatch());
    command.addOption("-hmode", "dssp");
    command.addOption("-nb");
    command.addOption("-cutoff", std::get<1>(params));
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestDSSPNB,
                         ::testing::Combine(::testing::Values("1cos.pdb",
                                                              "1hlc.pdb",
                                                              "1vzj.pdb",
                                                              "3byc.pdb",
                                                              "3kyy.pdb",
                                                              "4r80.pdb",
                                                              "4xjf.pdb",
                                                              "5u5p.pdb",
                                                              "7wgh.pdb",
                                                              "1gmc.pdb",
                                                              "1v3y.pdb",
                                                              "1yiw.pdb",
                                                              "2os3.pdb",
                                                              "3u04.pdb",
                                                              "4r6c.pdb",
                                                              "4wxl.pdb",
                                                              "5cvq.pdb",
                                                              "5i2b.pdb",
                                                              "5t8z.pdb",
                                                              "6jet.pdb"),
                                            ::testing::Values(0.9, 2.0)));



// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestGROMACSNB, )
{
    auto              params    = GetParam();
    const char* const cmdline[] = { "dssp" };
    std::string       fin(std::get<0>(params));
    // replace pdb in filename with dat to construct uniq output names
    std::string finname = std::regex_replace(fin, std::regex("\\.pdb"), "");
    CommandLine command(cmdline);
    setTopology(fin.c_str());
    setTrajectory(fin.c_str());
    setOutputFile("-o", formatString("%s-gromacs-nb-%.1f.dat", finname.c_str(), std::get<1>(params)).c_str(), ExactTextMatch());
    command.addOption("-hmode", "gromacs");
    command.addOption("-nb");
    command.addOption("-cutoff", std::get<1>(params));
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestGROMACSNB,
                         ::testing::Combine(::testing::Values("hdac.pdb",
                                                              "RNAseA.pdb",
                                                              "zyncfinger.pdb"),
                                            ::testing::Values(0.9, 2.0)));

// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestDSSPNoNB, )
{
    auto              params    = GetParam();
    const char* const cmdline[] = { "dssp" };
    std::string       fin(params);
    // replace pdb in filename with dat to construct uniq output names
    std::string finname = std::regex_replace(fin, std::regex("\\.pdb"), "");
    CommandLine command(cmdline);
    setTopology(fin.c_str());
    setTrajectory(fin.c_str());
    setOutputFile("-o", formatString("%s-dssp-nonb.dat", finname.c_str()).c_str(), ExactTextMatch());
    command.addOption("-hmode", "dssp");
    command.addOption("-nonb");
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestDSSPNoNB,
                         ::testing::Values("1cos.pdb",
                                           "1hlc.pdb",
                                           "1vzj.pdb",
                                           "3byc.pdb",
                                           "3kyy.pdb",
                                           "4r80.pdb",
                                           "4xjf.pdb",
                                           "5u5p.pdb",
                                           "7wgh.pdb",
                                           "1gmc.pdb",
                                           "1v3y.pdb",
                                           "1yiw.pdb",
                                           "2os3.pdb",
                                           "3u04.pdb",
                                           "4r6c.pdb",
                                           "4wxl.pdb",
                                           "5cvq.pdb",
                                           "5i2b.pdb",
                                           "5t8z.pdb",
                                           "6jet.pdb"));



// NOLINTNEXTLINE(cppcoreguidelines-owning-memory)
TEST_P(DsspModuleTestGROMACSNoNB, )
{
    auto              params    = GetParam();
    const char* const cmdline[] = { "dssp" };
    std::string       fin(params);
    // replace pdb in filename with dat to construct uniq output names
    std::string finname = std::regex_replace(fin, std::regex("\\.pdb"), "");
    CommandLine command(cmdline);
    setTopology(fin.c_str());
    setTrajectory(fin.c_str());
    setOutputFile("-o", formatString("%s-gromacs-nonb.dat", finname.c_str()).c_str(), ExactTextMatch());
    command.addOption("-hmode", "gromacs");
    command.addOption("-nonb");
    runTest(command);
}

INSTANTIATE_TEST_SUITE_P(MoleculeTests,
                         DsspModuleTestGROMACSNoNB,
                         ::testing::Values("hdac.pdb",
                                           "RNAseA.pdb",
                                           "zyncfinger.pdb"));


//    command.addOption("-nonb");
//    setOutputFile("-o", formatString("%s-%s-nonb.dat", finname.c_str(), std::get<2>(params)).c_str(), ExactTextMatch());
} // namespace
} // namespace test
} // namespace gmx
