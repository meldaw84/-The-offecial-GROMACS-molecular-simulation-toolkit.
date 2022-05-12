/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2010- The GROMACS Authors
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
 * Implements gmx::TrajectoryAnalysisRunnerCommon.
 *
 * \author Teemu Murtola <teemu.murtola@gmail.com>
 * \ingroup module_trajectoryanalysis
 */
#include "gmxpre.h"

#include "runnercommon.h"

#include <cstring>

#include <algorithm>
#include <string>
#include <vector>

#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/timecontrol.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/math/vec.h"
#include "gromacs/options/basicoptions.h"
#include "gromacs/options/filenameoption.h"
#include "gromacs/options/ioptionscontainer.h"
#include "gromacs/options/optionsvisitor.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/selection/selection.h"
#include "gromacs/selection/selectioncollection.h"
#include "gromacs/selection/selectionoption.h"
#include "gromacs/selection/selectionoptionbehavior.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/trajectoryanalysis/analysissettings.h"
#include "gromacs/trajectoryanalysis/topologyinformation.h"
#include "gromacs/utility/cstringutil.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/programcontext.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

#include "analysissettings_impl.h"

namespace gmx
{

//! Helper that holds all information associate with a single trajectory file
class InputData
{
public:
    InputData() : bTrajOpen_(false), fr_(nullptr), gpbc_(nullptr), status_(nullptr), oenv_(nullptr)
    {
    }
    ~InputData();

    InputData& operator=(const InputData& other) = default;
    InputData& operator=(InputData&& other) = default;
    InputData(const InputData& other)       = default;
    InputData(InputData&& other)            = default;

    void finishTrajectory();
    //! Name of particular input file.
    std::string fileName_;
    //! Is the current file open?
    bool bTrajOpen_;
    //! The current frame, or \p NULL if no frame loaded yet.
    t_trxframe* fr_;
    //! PBC removal stuff.
    gmx_rmpbc_t gpbc_;
    //! Used to store the status variable from read_first_frame().
    t_trxstatus* status_;
    //! Output env.
    gmx_output_env_t* oenv_;
};

void InputData::finishTrajectory()
{
    if (bTrajOpen_)
    {
        close_trx(status_);
        bTrajOpen_ = false;
    }
    if (gpbc_ != nullptr)
    {
        gmx_rmpbc_done(gpbc_);
        gpbc_ = nullptr;
    }
}


InputData::~InputData()
{
    finishTrajectory();
    if (fr_ != nullptr)
    {
        // There doesn't seem to be a function for freeing frame data
        sfree(fr_->x);
        sfree(fr_->v);
        sfree(fr_->f);
        sfree(fr_->index);
        sfree(fr_);
    }
    if (oenv_ != nullptr)
    {
        output_env_done(oenv_);
    }
}


class TrajectoryAnalysisRunnerCommon::Impl : public ITopologyProvider
{
public:
    explicit Impl(TrajectoryAnalysisSettings* settings);
    ~Impl() override;

    bool hasTrajectoryCollection() const { return !trjfiles_.empty(); }
    bool hasTrajectory() const
    {
        return hasTrajectoryCollection() && !trajectoryInputData_.fileName_.empty();
    }
    bool hasAnyTrajectory() const
    {
        return hasTrajectoryCollection()
               && std::any_of(trjfiles_.begin(), trjfiles_.end(), [](const auto& file) {
                      return !file.empty();
                  });
    }
    bool advanceTrajectory();

    int trajectoryCollectionSize() const { return trjfiles_.size(); }

    void initTopology(bool required);
    void initTrajectoryCollection();
    void initFirstFrameOfTrajectory();
    void initFrameIndexGroup();
    void finishTrajectory();

    // From ITopologyProvider
    gmx_mtop_t* getTopology(bool required) override
    {
        initTopology(required);
        return topInfo_.mtop_.get();
    }
    int getAtomCount() override
    {
        if (!topInfo_.hasTopology())
        {
            if (trajectoryGroup_.isValid())
            {
                GMX_THROW(InconsistentInputError(
                        "-fgroup is only supported when -s is also specified"));
            }
            // Read the first frame if we don't know the maximum number of
            // atoms otherwise. Use the current index of files from array of inputs.
            initTrajectoryCollection();
            initFirstFrameOfTrajectory();
            return trajectoryInputData_.fr_->natoms;
        }
        return -1;
    }

    TrajectoryAnalysisSettings& settings_;
    TopologyInformation         topInfo_;

    //! Name of the trajectory file (empty if not provided).
    std::vector<std::string> trjfiles_;
    //! Index into array of trajectory files.
    int trjFileIndex_;
    //! Name of the topology file (empty if no topology provided).
    std::string topfile_;
    Selection   trajectoryGroup_;
    double      startTime_;
    double      endTime_;
    double      deltaTime_;
    bool        bStartTimeSet_;
    bool        bEndTimeSet_;
    bool        bDeltaTimeSet_;
    bool        bAllowMultipleInputs_;
    //! Collection of data related to input trajectory files.
    InputData trajectoryInputData_;
};


TrajectoryAnalysisRunnerCommon::Impl::Impl(TrajectoryAnalysisSettings* settings) :
    settings_(*settings),
    trjFileIndex_(0),
    startTime_(0.0),
    endTime_(0.0),
    deltaTime_(0.0),
    bStartTimeSet_(false),
    bEndTimeSet_(false),
    bDeltaTimeSet_(false),
    bAllowMultipleInputs_(false)
{
}


TrajectoryAnalysisRunnerCommon::Impl::~Impl() {}

void TrajectoryAnalysisRunnerCommon::Impl::initTrajectoryCollection()
{
    // No files present, no need to do anything
    if (trjfiles_.empty())
    {
        return;
    }
    if (trajectoryCollectionSize() < trjFileIndex_)
    {
        GMX_THROW(InternalError("Index into trajectory collection out of range"));
    }
    trajectoryInputData_.fileName_ = trjfiles_[trjFileIndex_];
}

void TrajectoryAnalysisRunnerCommon::Impl::initTopology(bool required)
{
    // Return immediately if the topology has already been loaded.
    if (topInfo_.hasTopology())
    {
        return;
    }

    if (required && topfile_.empty())
    {
        GMX_THROW(InconsistentInputError("No topology provided, but one is required for analysis"));
    }

    // Load the topology if requested.
    if (!topfile_.empty())
    {
        topInfo_.fillFromInputFile(topfile_);
        initTrajectoryCollection();
        if (hasTrajectory() && !settings_.hasFlag(TrajectoryAnalysisSettings::efUseTopX))
        {
            topInfo_.xtop_.clear();
        }
        if (hasTrajectory() && !settings_.hasFlag(TrajectoryAnalysisSettings::efUseTopV))
        {
            topInfo_.vtop_.clear();
        }
    }
}

void TrajectoryAnalysisRunnerCommon::Impl::initFirstFrameOfTrajectory()
{
    // Return if we have already initialized the trajectory and are still on the same file
    // in the array of input files.
    if (trajectoryInputData_.fr_ != nullptr)
    {
        return;
    }
    output_env_init(
            &trajectoryInputData_.oenv_, getProgramContext(), settings_.timeUnit(), FALSE, XvgFormat::None, 0);

    int frflags = settings_.frflags();
    frflags |= TRX_NEED_X;

    snew(trajectoryInputData_.fr_, 1);

    if (hasTrajectory())
    {
        if (!read_first_frame(trajectoryInputData_.oenv_,
                              &trajectoryInputData_.status_,
                              trajectoryInputData_.fileName_.c_str(),
                              trajectoryInputData_.fr_,
                              frflags))
        {
            GMX_THROW(FileIOError("Could not read coordinates from trajectory"));
        }
        trajectoryInputData_.bTrajOpen_ = true;

        if (topInfo_.hasTopology())
        {
            const int topologyAtomCount = topInfo_.mtop()->natoms;
            if (trajectoryInputData_.fr_->natoms > topologyAtomCount)
            {
                const std::string message =
                        formatString("Trajectory (%d atoms) does not match topology (%d atoms)",
                                     trajectoryInputData_.fr_->natoms,
                                     topologyAtomCount);
                GMX_THROW(InconsistentInputError(message));
            }
        }
    }
    else
    {
        // Prepare a frame from topology information.
        if (frflags & (TRX_NEED_F))
        {
            GMX_THROW(InvalidInputError("Forces cannot be read from a topology"));
        }
        trajectoryInputData_.fr_->natoms = topInfo_.mtop()->natoms;
        trajectoryInputData_.fr_->bX     = TRUE;
        snew(trajectoryInputData_.fr_->x, trajectoryInputData_.fr_->natoms);
        memcpy(trajectoryInputData_.fr_->x,
               topInfo_.xtop_.data(),
               sizeof(*trajectoryInputData_.fr_->x) * trajectoryInputData_.fr_->natoms);
        if (frflags & (TRX_NEED_V))
        {
            if (topInfo_.vtop_.empty())
            {
                GMX_THROW(InvalidInputError(
                        "Velocities were required, but could not be read from the topology file"));
            }
            trajectoryInputData_.fr_->bV = true;
            snew(trajectoryInputData_.fr_->v, trajectoryInputData_.fr_->natoms);
            memcpy(trajectoryInputData_.fr_->v,
                   topInfo_.vtop_.data(),
                   sizeof(*trajectoryInputData_.fr_->v) * trajectoryInputData_.fr_->natoms);
        }
        trajectoryInputData_.fr_->bBox = true;
        copy_mat(topInfo_.boxtop_, trajectoryInputData_.fr_->box);
    }

    setTrxFramePbcType(trajectoryInputData_.fr_, topInfo_.pbcType());
    if (topInfo_.hasTopology() && settings_.hasRmPBC())
    {
        trajectoryInputData_.gpbc_ = gmx_rmpbc_init(topInfo_);
    }
}

void TrajectoryAnalysisRunnerCommon::Impl::initFrameIndexGroup()
{
    if (!trajectoryGroup_.isValid())
    {
        return;
    }
    GMX_RELEASE_ASSERT(trajectoryInputData_.bTrajOpen_,
                       "Trajectory index only makes sense with a real trajectory");
    if (trajectoryGroup_.atomCount() != trajectoryInputData_.fr_->natoms)
    {
        const std::string message = formatString(
                "Selection specified with -fgroup has %d atoms, but "
                "the trajectory (-f) has %d atoms.",
                trajectoryGroup_.atomCount(),
                trajectoryInputData_.fr_->natoms);
        GMX_THROW(InconsistentInputError(message));
    }
    trajectoryInputData_.fr_->bIndex = TRUE;
    snew(trajectoryInputData_.fr_->index, trajectoryGroup_.atomCount());
    std::copy(trajectoryGroup_.atomIndices().begin(),
              trajectoryGroup_.atomIndices().end(),
              trajectoryInputData_.fr_->index);
}

bool TrajectoryAnalysisRunnerCommon::Impl::advanceTrajectory()
{
    ++trjFileIndex_;
    bool inRange = trjFileIndex_ < trajectoryCollectionSize();
    if (inRange)
    {
        trajectoryInputData_.fileName_ = trjfiles_[trjFileIndex_];
        initFirstFrameOfTrajectory();
        initFrameIndexGroup();
    }
    return inRange;
}

void TrajectoryAnalysisRunnerCommon::Impl::finishTrajectory()
{
    trajectoryInputData_ = InputData();
}

/*********************************************************************
 * TrajectoryAnalysisRunnerCommon
 */

TrajectoryAnalysisRunnerCommon::TrajectoryAnalysisRunnerCommon(TrajectoryAnalysisSettings* settings) :
    impl_(new Impl(settings))
{
}


TrajectoryAnalysisRunnerCommon::~TrajectoryAnalysisRunnerCommon() {}


ITopologyProvider* TrajectoryAnalysisRunnerCommon::topologyProvider()
{
    return impl_.get();
}


void TrajectoryAnalysisRunnerCommon::initOptions(IOptionsContainer* options, TimeUnitBehavior* timeUnitBehavior)
{
    TrajectoryAnalysisSettings& settings = impl_->settings_;

    // Add common file name arguments.
    if (impl_->bAllowMultipleInputs_)
    {
        options->addOption(
                FileNameOption("f")
                        .filetype(OptionFileType::Trajectory)
                        .inputFile()
                        .storeVector(&impl_->trjfiles_)
                        .multiValue()
                        .defaultBasename("traj")
                        .description("Set of input trajectories or single configuration"));
    }
    else
    {
        // fix vector size at 1 and use only the first field
        impl_->trjfiles_.resize(1);
        options->addOption(FileNameOption("f")
                                   .filetype(OptionFileType::Trajectory)
                                   .inputFile()
                                   .store(&impl_->trjfiles_[0])
                                   .defaultBasename("traj")
                                   .description("Input trajectory or single configuration"));
    }
    options->addOption(FileNameOption("s")
                               .filetype(OptionFileType::Topology)
                               .inputFile()
                               .store(&impl_->topfile_)
                               .defaultBasename("topol")
                               .description("Input structure"));

    // Add options for trajectory time control.
    options->addOption(DoubleOption("b")
                               .store(&impl_->startTime_)
                               .storeIsSet(&impl_->bStartTimeSet_)
                               .timeValue()
                               .description("First frame (%t) to read from trajectory"));
    options->addOption(DoubleOption("e")
                               .store(&impl_->endTime_)
                               .storeIsSet(&impl_->bEndTimeSet_)
                               .timeValue()
                               .description("Last frame (%t) to read from trajectory"));
    options->addOption(DoubleOption("dt")
                               .store(&impl_->deltaTime_)
                               .storeIsSet(&impl_->bDeltaTimeSet_)
                               .timeValue()
                               .description("Only use frame if t MOD dt == first time (%t)"));

    // Add time unit option.
    timeUnitBehavior->setTimeUnitFromEnvironment();
    timeUnitBehavior->addTimeUnitOption(options, "tu");
    timeUnitBehavior->setTimeUnitStore(&impl_->settings_.impl_->timeUnit);

    options->addOption(SelectionOption("fgroup")
                               .store(&impl_->trajectoryGroup_)
                               .onlySortedAtoms()
                               .onlyStatic()
                               .description("Atoms stored in the trajectory file "
                                            "(if not set, assume first N atoms)"));

    // Add plot options.
    settings.impl_->plotSettings.initOptions(options);

    // Add common options for trajectory processing.
    if (!settings.hasFlag(TrajectoryAnalysisSettings::efNoUserRmPBC))
    {
        options->addOption(
                BooleanOption("rmpbc").store(&settings.impl_->bRmPBC).description("Make molecules whole for each frame"));
    }
    if (!settings.hasFlag(TrajectoryAnalysisSettings::efNoUserPBC))
    {
        options->addOption(
                BooleanOption("pbc")
                        .store(&settings.impl_->bPBC)
                        .description("Use periodic boundary conditions for distance calculation"));
    }
}


void TrajectoryAnalysisRunnerCommon::optionsFinished()
{
    if (!impl_->hasAnyTrajectory() && impl_->topfile_.empty())
    {
        GMX_THROW(InconsistentInputError("No trajectory or topology provided, nothing to do!"));
    }

    if (impl_->trajectoryGroup_.isValid() && !impl_->hasAnyTrajectory())
    {
        GMX_THROW(
                InconsistentInputError("-fgroup only makes sense together with a trajectory (-f)"));
    }

    impl_->settings_.impl_->plotSettings.setTimeUnit(impl_->settings_.timeUnit());

    if (impl_->bStartTimeSet_)
    {
        setTimeValue(TimeControl::Begin, impl_->startTime_);
    }
    if (impl_->bEndTimeSet_)
    {
        setTimeValue(TimeControl::End, impl_->endTime_);
    }
    if (impl_->bDeltaTimeSet_)
    {
        setTimeValue(TimeControl::Delta, impl_->deltaTime_);
    }
}

void TrajectoryAnalysisRunnerCommon::allowMultipleInputTrajectories()
{
    impl_->bAllowMultipleInputs_ = true;
}

void TrajectoryAnalysisRunnerCommon::initTopology()
{
    const bool topologyRequired = impl_->settings_.hasFlag(TrajectoryAnalysisSettings::efRequireTop);
    impl_->initTopology(topologyRequired);
}


void TrajectoryAnalysisRunnerCommon::initFirstFrame()
{
    impl_->initFirstFrameOfTrajectory();
}


void TrajectoryAnalysisRunnerCommon::initFrameIndexGroup()
{
    impl_->initFrameIndexGroup();
}


bool TrajectoryAnalysisRunnerCommon::readNextFrame()
{
    bool bContinue = false;
    if (hasTrajectory())
    {
        bContinue = read_next_frame(impl_->trajectoryInputData_.oenv_,
                                    impl_->trajectoryInputData_.status_,
                                    impl_->trajectoryInputData_.fr_);
    }
    if (!bContinue)
    {
        impl_->finishTrajectory();
        bContinue = impl_->advanceTrajectory();
    }
    return bContinue;
}


void TrajectoryAnalysisRunnerCommon::initFrame()
{
    if (impl_->trajectoryInputData_.gpbc_ != nullptr)
    {
        gmx_rmpbc_trxfr(impl_->trajectoryInputData_.gpbc_, impl_->trajectoryInputData_.fr_);
    }
}


bool TrajectoryAnalysisRunnerCommon::hasTrajectory() const
{
    return impl_->hasTrajectory();
}


const TopologyInformation& TrajectoryAnalysisRunnerCommon::topologyInformation() const
{
    return impl_->topInfo_;
}


t_trxframe& TrajectoryAnalysisRunnerCommon::frame() const
{
    GMX_RELEASE_ASSERT(impl_->trajectoryInputData_.fr_ != nullptr,
                       "Frame not available when accessed");
    return *impl_->trajectoryInputData_.fr_;
}

} // namespace gmx
