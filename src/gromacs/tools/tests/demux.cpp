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
 * Tests for functionality of the "demux" tool.
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 */
#include "gmxpre.h"

#include "gromacs/tools/demux.h"

#include <vector>

#include <googletest/googletest/include/gtest/gtest.h>
#include <gtest/gtest-param-test.h>

#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textwriter.h"

#include "testutils/cmdlinetest.h"
#include "testutils/testfilemanager.h"

namespace gmx
{

namespace test
{

//! Convienence type for input related data.
using DemuxInputParams = std::tuple<std::string, std::vector<std::string>>;
//! Handle for holding input data for demuxing.
using DemuxTestParams = std::tuple<DemuxInputParams, std::string, bool>;

class DemuxTest : public ::testing::Test, public ::testing::WithParamInterface<DemuxTestParams>
{
public:
    //! Run test case.
    void runTest(CommandLine* cmdline, const DemuxInputParams& params, bool useFileList);
    TestFileManager* manager() { return &manager_; }

private:
    TestFileManager manager_;
};

namespace
{
//! Helper to write text file with input files for demuxing tool, needed for getting correct test paths.
std::string writeTestInputFile(gmx::ArrayRef<const std::string> fileList, TestFileManager* manager)
{
    std::string     outputFile = manager->getTemporaryFilePath("demux.txt");
    gmx::TextWriter writer(outputFile);
    for (const auto& file : fileList)
    {
        writer.writeLine(manager->getInputFilePath(file));
    }
    writer.close();
    return outputFile;
}
} // namespace

void DemuxTest::runTest(CommandLine* cmdline, const DemuxInputParams& params, bool useFileList)
{
    cmdline->addOption("-input", manager()->getInputFilePath(std::get<0>(params)));
    if (useFileList)
    {
        cmdline->addOption("-filelist", writeTestInputFile(std::get<1>(params), manager()));
    }
    else
    {
        cmdline->append("-f");
        for (const auto& name : std::get<1>(params))
        {
            cmdline->append(manager()->getInputFilePath(name));
        }
    }
    printf("%s\n", cmdline->toString().c_str());
    EXPECT_EQ(0, gmx::test::CommandLineTestHelper::runModuleFactory(&gmx::DemuxInfo::create, cmdline));
}

TEST_P(DemuxTest, WorksForWholeFile)
{
    auto              params    = GetParam();
    const char* const command[] = { "demux" };
    CommandLine       cmdline(command);
    cmdline.addOption("-o", std::get<1>(params));
    runTest(&cmdline, std::get<0>(params), std::get<2>(params));
}

TEST_P(DemuxTest, WorksWithStartTime)
{
    auto              params    = GetParam();
    const char* const command[] = { "demux" };
    CommandLine       cmdline(command);
    cmdline.addOption("-o", std::get<1>(params));
    cmdline.addOption("-b", "1.5");
    runTest(&cmdline, std::get<0>(params), std::get<2>(params));
}

TEST_P(DemuxTest, WorksWithEndTime)
{
    auto              params    = GetParam();
    const char* const command[] = { "demux" };
    CommandLine       cmdline(command);
    cmdline.addOption("-o", std::get<1>(params));
    cmdline.addOption("-e", "3.5");
    runTest(&cmdline, std::get<0>(params), std::get<2>(params));
}

TEST_P(DemuxTest, WorksWithStartAndEndTime)
{
    auto              params    = GetParam();
    const char* const command[] = { "demux" };
    CommandLine       cmdline(command);
    cmdline.addOption("-o", std::get<1>(params));
    cmdline.addOption("-b", "1.5");
    cmdline.addOption("-e", "3.5");
    runTest(&cmdline, std::get<0>(params), std::get<2>(params));
}

const DemuxInputParams twoFiles  = { "demux1.xvg", { "demux1_1.pdb", "demux1_2.pdb" } };
const DemuxInputParams fourFiles = {
    "demux2.xvg",
    { "demux2_1.pdb", "demux2_2.pdb", "demux2_3.pdb", "demux2_4.pdb" }
};

INSTANTIATE_TEST_SUITE_P(ToolWorks,
                         DemuxTest,
                         ::testing::Combine(::testing::Values(twoFiles, fourFiles),
                                            ::testing::Values("test.trr", "test.xtc"),
                                            ::testing::Values(false, true)));


} // namespace test

} // namespace gmx
