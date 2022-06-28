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
 * \brief Implements gmx demux utility.
 *
 * This is separate from the other trajectory analysis tools, as the demux tool
 * needs to read in several trajectory files at the same time (instead of looping
 * over several files in order). So we add custom code her to handle this kind
 * of behaviour instead of re-using the exisiting TAF.
 *
 * \ingroup module_tools
 */
#include "gmxpre.h"

#include "demux.h"

#include "config.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <optional>
#include <vector>

#include "gromacs/commandline/cmdlineoptionsmodule.h"
#include "gromacs/coordinateio/coordinatefile.h"
#include "gromacs/coordinateio/requirements.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/timecontrol.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/selection/selection.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/filestream.h"
#include "gromacs/utility/path.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/stringstream.h"
#include "gromacs/utility/stringutil.h"
#include "gromacs/utility/textreader.h"

namespace gmx
{

namespace
{

//! Initialize all input trajectory files.
std::vector<t_trxframe*> initializeAllTrajectoryFiles(gmx::ArrayRef<const std::string> fileNames,
                                                      gmx::ArrayRef<t_trxstatus*>      fileStatus,
                                                      gmx_output_env_t*                oenv,
                                                      std::optional<int64_t>           atomNumber)
{
    std::vector<t_trxframe*> frames(fileNames.size(), nullptr);
    for (auto& frame : frames)
    {
        snew(frame, 1);
    }
    int frflags = TRX_NEED_X;
    for (int index = 0; index < gmx::ssize(fileNames); ++index)
    {
        if (!read_first_frame(oenv, &fileStatus[index], fileNames[index].c_str(), frames[index], frflags))
        {
            GMX_THROW(FileIOError(gmx::formatString(
                    "Could not read coordinates from trajectory file, index %d", index)));
        }
        if (!atomNumber.has_value())
        {
            atomNumber = frames[index]->natoms;
        }
        if (frames[index]->natoms != *atomNumber)
        {
            GMX_THROW(InvalidInputError(
                    gmx::formatString("Number of atoms in frame %d (%d) doesn't match the atom "
                                      "number in the topology %ld",
                                      index,
                                      frames[index]->natoms,
                                      *atomNumber)));
        }
    }
    return frames;
}
//! Advance trajectory collection together.
bool readNextFrameOfAllFiles(gmx_output_env_t*           oenv,
                             gmx::ArrayRef<t_trxstatus*> fileStatus,
                             gmx::ArrayRef<t_trxframe*>  frames)
{
    bool framesAreValid = true;
    for (int index = 0; index < gmx::ssize(frames) && framesAreValid; ++index)
    {
        framesAreValid = read_next_frame(oenv, fileStatus[index], frames[index]);
    }
    return framesAreValid;
}

//! Clean up all files together.
void cleanupAllTrajectoryFiles(gmx::ArrayRef<t_trxstatus*> fileStatus, gmx::ArrayRef<t_trxframe*> frames)
{
    for (int index = 0; index < gmx::ssize(frames); ++index)
    {
        done_frame(frames[index]);
        close_trx(fileStatus[index]);
    }
}

//! Check that all times in the \p frames match
real checkFrameConsistency(gmx::ArrayRef<t_trxframe*> frames)
{
    std::optional<real> frameTime;
    for (int index = 0; index < gmx::ssize(frames); ++index)
    {
        if (!frameTime.has_value())
        {
            frameTime = frames[index]->time;
        }
        if (std::round(*frameTime - frames[index]->time) != 0)
        {
            GMX_THROW(InconsistentInputError(
                    gmx::formatString("The time read in from trajectory %d (%3.8f) does not match "
                                      "the time read in from first trajectory (%3.8f)",
                                      index,
                                      frames[index]->time,
                                      *frameTime)));
        }
    }
    return *frameTime;
}

//! Check that time values are consistent.
void checkTimeConsistency(real frameTime, real demuxTime)
{
    if (std::round(frameTime - demuxTime) != 0)
    {
        GMX_THROW(InconsistentInputError(
                gmx::formatString("The time read in from trajectories (%3.8f) does not match the "
                                  "time read in from demuxing file (%3.8f)",
                                  frameTime,
                                  demuxTime)));
    }
}

//! Read in input file containing file list and convert to files to open
std::vector<std::string> readInputFileIntoFileList(const std::string& trajectoryListFileName)
{
    gmx::TextReader          reader(trajectoryListFileName);
    std::vector<std::string> fileList;
    std::string              line;
    while (reader.readLine(&line))
    {
        fileList.emplace_back(gmx::stripString(gmx::Path::normalize(line)));
    }
    return fileList;
}

class Demux : public ICommandLineOptionsModule
{
public:
    Demux() {}
    ~Demux() override;

    void init(CommandLineModuleSettings* /*settings*/) override {}

    void initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings) override;

    void optionsFinished() override;

    int run() override;

private:
    //! Input TPR file name, optional.
    std::string inputTprFilename_;
    //! List of input trajectory file names.
    std::vector<std::string> trajectoryFileNames_;
    //! Name of single input file to read trajectory file names from.
    std::string trajectoryListFileName_;
    //! Input file with demux information.
    std::string demuxIndexFileName_;
    bool        haveInputTpr_ = false;
    //! Do we have read in a vector of input files directly.
    bool haveReadInVectorOfFileNames_ = false;
    //! Do we have a list of input trajectory files.
    bool haveTrajectoryFileList_ = false;
    //! Output environment storage.
    gmx_output_env_t* oenv_ = nullptr;
    //! Use first time to start analysis from.
    bool startTimeIsSet_ = false;
    //! Use last time to analyse.
    bool endTimeIsSet_ = false;
    //! First time to analyse.
    double startTime_;
    //! Last time to analyse.
    double endTime_;
    //! Time unit to use.
    TimeUnit timeUnit_;
    //! Storage of requirements for creating output files.
    OutputRequirementOptionDirector requirementsBuilder_;
    //! Name for output file.
    std::string outputNamePrefix_;
};

Demux::~Demux()
{
    if (oenv_ != nullptr)
    {
        output_env_done(oenv_);
    }
}

void Demux::initOptions(IOptionsContainer* options, ICommandLineOptionsModuleSettings* settings)
{
    const char* desc[] = {
        "[THISMODULE] allows rematching the contents of trajectory files from a replica exchange "
        "simulation.",
        "By reading in a collection of trajectory ([REF].trr[ref]/[REF].xtc[ref]/[TT]tng[tt]) "
        "files and",
        "an index ([REF].xvg[ref]) file, the input frames are matched to the correct trajectory "
        "according",
        "to the initial configurations."
    };
    settings->setHelpText(desc);
    TimeUnitBehavior timeUnitBehavior;

    const char* bugs[] = {
        "If the demuxing index file is incomplete, the tool can't know how to assign the correct "
        "frames and will likely do something nonsensical."
    };
    settings->setBugText(bugs);

    options->addOption(FileNameOption("s")
                               .filetype(OptionFileType::RunInput)
                               .inputFile()
                               .store(&inputTprFilename_)
                               .storeIsSet(&haveInputTpr_)
                               .defaultBasename("topol")
                               .description("Run input file to dump"));
    options->addOption(
            FileNameOption("f")
                    .multiValue()
                    .filetype(OptionFileType::Trajectory)
                    .inputFile()
                    .storeVector(&trajectoryFileNames_)
                    .storeIsSet(&haveReadInVectorOfFileNames_)
                    .description("Trajectory files to demux, read in directly from command line"));
    options->addOption(
            FileNameOption("filelist")
                    .filetype(OptionFileType::RawText)
                    .inputFile()
                    .store(&trajectoryListFileName_)
                    .storeIsSet(&haveTrajectoryFileList_)
                    .defaultBasename("filelist")
                    .description("Input file containing list of trajectory files to demux"));
    options->addOption(
            FileNameOption("o")
                    .filetype(OptionFileType::Trajectory)
                    .outputFile()
                    .store(&outputNamePrefix_)
                    .defaultBasename("trajout")
                    .required()
                    .description("Prefix for the name of the trajectory files written."));
    options->addOption(FileNameOption("input")
                               .filetype(OptionFileType::Plot)
                               .inputFile()
                               .required()
                               .store(&demuxIndexFileName_)
                               .description("Index file containing list of frame times and index "
                                            "matching in Xvg format"));
    // Add options for trajectory time control.
    options->addOption(
            DoubleOption("b")
                    .store(&startTime_)
                    .storeIsSet(&startTimeIsSet_)
                    .timeValue()
                    .description("First frame (%t) to read and analyse from trajectory"));
    options->addOption(DoubleOption("e")
                               .store(&endTime_)
                               .storeIsSet(&endTimeIsSet_)
                               .timeValue()
                               .description("Last frame (%t) to read and analyse from trajectory"));
    // Add time unit option.
    timeUnitBehavior.setTimeUnitFromEnvironment();
    timeUnitBehavior.addTimeUnitOption(options, "tu");
    timeUnitBehavior.setTimeUnitStore(&timeUnit_);
    // How to write output files.
    requirementsBuilder_.initOptions(options);
}

void Demux::optionsFinished()
{
    if ((haveReadInVectorOfFileNames_ && haveTrajectoryFileList_)
        || (!haveReadInVectorOfFileNames_ && !haveTrajectoryFileList_))
    {
        GMX_THROW(InconsistentInputError(
                "Need to have read in either a number of files from command line, or a single file "
                "containing a list of files to demux"));
    }
    if (haveTrajectoryFileList_)
    {
        trajectoryFileNames_ = readInputFileIntoFileList(trajectoryListFileName_);
    }
    if (trajectoryFileNames_.size() == 1)
    {
        GMX_THROW(InvalidInputError(
                "This tool only works for several trajectory files, not a single file"));
    }
    if (startTimeIsSet_)
    {
        setTimeValue(TimeControl::Begin, startTime_);
    }
    if (endTimeIsSet_)
    {
        setTimeValue(TimeControl::End, endTime_);
    }
}

int Demux::run()
{
    TopologyInformation topology;
    if (haveInputTpr_)
    {
        topology.fillFromInputFile(inputTprFilename_);
    }
    const int           numberOfFiles = trajectoryFileNames_.size();
    std::optional<real> startTime     = startTimeIsSet_ ? std::optional(startTime_) : std::nullopt;
    std::optional<real> endTime       = endTimeIsSet_ ? std::optional(endTime_) : std::nullopt;
    auto demuxInformation             = readXvgTimeSeries(demuxIndexFileName_, startTime, endTime);
    if (numberOfFiles != demuxInformation.extent(1) - 1)
    {
        GMX_THROW(InconsistentInputError(
                "Number of files doesn't match number of columns in demuxing file"));
    }
    std::vector<TrajectoryFrameWriterPointer> writers;
    for (int index = 0; index < numberOfFiles; ++index)
    {
        std::string outputName =
                Path::concatenateBeforeExtension(outputNamePrefix_, formatString("_%d", index));
        writers.emplace_back(createTrajectoryFrameWriter(topology.mtop(),
                                                         {},
                                                         outputName,
                                                         topology.hasTopology() ? topology.copyAtoms() : nullptr,
                                                         requirementsBuilder_.process()));
    }

    output_env_init(&oenv_, getProgramContext(), timeUnit_, FALSE, XvgFormat::None, 0);
    std::vector<t_trxstatus*> frameStatus(numberOfFiles, nullptr);
    std::vector<t_trxframe*>  frames = initializeAllTrajectoryFiles(
            trajectoryFileNames_,
            frameStatus,
            oenv_,
            haveInputTpr_ ? std::optional(topology.mtop()->natoms) : std::nullopt);

    // The demux table that has been read in starts after startTime and ends before endTime, so
    // any row we read from it starts indexing from 0 at this point.
    int frameIndex = 0;
    do
    {
        const real frameTime = checkFrameConsistency(frames);
        // only try to do demuxing when we have the correct time.
        if ((!startTime.has_value() || frameTime > *startTime)
            && (!endTime.has_value() || frameTime < *endTime))
        {
            // The maximum number of rows in the file always needs to be larger or equal
            // to the number of frames to analyse
            if (frameIndex > demuxInformation.extent(0))
            {
                GMX_THROW(InconsistentInputError(
                        gmx::formatString("Number of frames for analysis (%d) is larger than the "
                                          "number of rows read from the demuxing table (%ld)",
                                          frameIndex,
                                          demuxInformation.extent(0))));
            }
            const auto viewOnDemuxAtIndex = demuxInformation.asConstView()[frameIndex];
            // First column in demux table is always the time
            checkTimeConsistency(frameTime, viewOnDemuxAtIndex[0]);
            for (int fileIndex = 0; fileIndex < numberOfFiles; ++fileIndex)
            {
                writers[static_cast<int>(viewOnDemuxAtIndex[fileIndex + 1])]->prepareAndWriteFrame(
                        frameIndex, *frames[fileIndex]);
            }
            ++frameIndex;
        }
    } while (readNextFrameOfAllFiles(oenv_, frameStatus, frames));
    cleanupAllTrajectoryFiles(frameStatus, frames);
    return 0;
}

} // namespace

LIBGROMACS_EXPORT const char DemuxInfo::name[] = "demux";
LIBGROMACS_EXPORT const char DemuxInfo::shortDescription[] =
        "Allow demuxing of trajectory files obtained from replica exchange simulations, moving "
        "frames to the correct corresponding trajectory file.";
ICommandLineOptionsModulePointer DemuxInfo::create()
{
    return std::make_unique<Demux>();
}

} // namespace gmx
