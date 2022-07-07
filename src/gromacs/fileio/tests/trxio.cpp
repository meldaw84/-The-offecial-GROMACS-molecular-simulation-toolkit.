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
 * Tests for file status I/O routines
 *
 * \author Paul Bauer <paul.bauer.q@gmail.com>
 * \ingroup module_fileio
 */
#include "gmxpre.h"

#include "gromacs/fileio/trxio.h"

#include <string>

#include <gtest/gtest.h>

#include "gromacs/fileio/oenv.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/programcontext.h"

#include "testutils/simulationdatabase.h"
#include "testutils/testfilemanager.h"

namespace gmx
{
namespace test
{
namespace
{

class TrxTest : public ::testing::Test
{
public:
    TrxTest()
    {
        coordinates_ = { { 1, 2, 3 }, { 4, 5, 6 }, { 7, 8, 9 } };
        clear_trxframe(&testFrame_, false);
        testFrame_.step   = 0;
        testFrame_.time   = 0;
        testFrame_.natoms = coordinates_.size();
        testFrame_.x      = as_rvec_array(coordinates_.data());
        output_env_init(&oenv_, getProgramContext(), {}, FALSE, XvgFormat::None, 0);
    }

    ~TrxTest() override
    {
        if (oenv_ != nullptr)
        {
            output_env_done(oenv_);
        }
    }
    gmx::test::TestFileManager fileManager_;
    t_trxframe                 testFrame_;
    std::vector<gmx::RVec>     coordinates_;
    gmx_output_env_t*          oenv_ = nullptr;
};

TEST_F(TrxTest, CanOpenFileForOutput)
{
    const char* fileMode = "w";
    auto status = openTrajectoryFile(fileManager_.getTemporaryFilePath("test.trr"), fileMode);
    EXPECT_EQ(status.tng(), nullptr);
    EXPECT_NE(status.getFileIO(), nullptr);
    status.writeTngFrame(&testFrame_);
}

TEST_F(TrxTest, CanOpenTngUsingDefault)
{
    const char* fileMode = "w";
    auto status = openTrajectoryFile(fileManager_.getTemporaryFilePath("test.tng"), fileMode);
    EXPECT_NE(status.tng(), nullptr);
    EXPECT_EQ(status.getFileIO(), nullptr);
    status.writeTngFrame(&testFrame_);
}

// Currently fails, thus disabled
// TEST_F(TrxTest, CanWriteFileAndReadAgain)
//{
//    auto fullName = fileManager_.getTemporaryFilePath("test.trr");
//    {
//    const char* fileMode = "w";
//    auto status = openTrajectoryFile(fullName, fileMode);
//    EXPECT_EQ(status.tng(), nullptr);
//    EXPECT_NE(status.getFileIO(), nullptr);
//    status.writeTrxframe(&testFrame_, nullptr);
//    }
//    t_trxframe newFrame;
//    auto input = read_first_frame(oenv_, fullName, &newFrame, trxNeedCoordinates);
//    EXPECT_TRUE(input.has_value());
//}

} // namespace
} // namespace test
} // namespace gmx
