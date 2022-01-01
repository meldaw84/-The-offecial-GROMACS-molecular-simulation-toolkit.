/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2013- The GROMACS Authors
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
 *  \brief
 *  Tool for extracting AWH (Accelerated Weight Histogram) data from energy files.
 *
 *  \author Viveca Lindahl
 *  \author Berk Hess
 */

#include "gmxpre.h"

#include <cstdlib>
#include <cstring>

#include <algorithm>
#include <array>
#include <memory>
#include <string>

#include "gromacs/applied_forces/awh/read_params.h"
#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/enxio.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/fileio/xvgr.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/math/units.h"
#include "gromacs/mdtypes/awh_params.h"
#include "gromacs/mdtypes/inputrec.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/energyframe.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/basedefinitions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"
#include "gromacs/utility/stringutil.h"

using gmx::AwhBiasParams;
using gmx::AwhParams;

namespace
{

//! Enum for choosing AWH graph output
enum class AwhGraphSelection
{
    Pmf, //!< Only the PMF
    All  //!< All possible AWH graphs
};

//! Energy unit options
enum class EnergyUnit
{
    KJPerMol, //!< kJ/mol
    KT        //!< kT
};

} // namespace

/*! \brief All meta-data that is shared for one output file type for one bias */
class OutputFile
{
public:
    /*! \brief Constructor, Set the output base file name and title.
     *
     * Result is a valid object, but will produce empty output files.
     *
     * \param[in] filename   The name for output files, frame time, and possibly bias number, will be added per file/frame.
     * \param[in] baseTitle  The base title of the plot, the bias number might be added.
     * \param[in] numBias    The total number of AWH biases in the system.
     * \param[in] biasIndex  The index of this bias.
     */
    OutputFile(const std::string& filename, const std::string& baseTitle, int numBias, int biasIndex);

    /*! \brief Initializes the output file setup for the AWH output.
     *
     * \param[in] subBlockStart   Index of the first sub-block to write in the energy frame.
     * \param[in] numSubBlocks    The total number of sub-blocks in the framw.
     * \param[in] awhBiasParams   The AWH bias parameters.
     * \param[in] graphSelection  Selects which graphs to plot.
     * \param[in] energyUnit      Requested energy unit in output.
     * \param[in] kTValue         kB*T in kJ/mol.
     */
    void initializeAwhOutputFile(int                  subBlockStart,
                                 int                  numSubBlocks,
                                 const AwhBiasParams& awhBiasParams,
                                 AwhGraphSelection    graphSelection,
                                 EnergyUnit           energyUnit,
                                 real                 kTValue);

    /*! \brief Initializes the output file setup for the fricion output.
     *
     * \param[in] subBlockStart   Index of the first sub-block to write in the energy frame.
     * \param[in] numSubBlocks    The total number of sub-blocks in the framw.
     * \param[in] awhBiasParams   The AWH bias parameters.
     * \param[in] energyUnit      Requested energy unit in output.
     * \param[in] kTValue         kB*T in kJ/mol.
     */
    void initializeFrictionOutputFile(int                  subBlockStart,
                                      int                  numSubBlocks,
                                      const AwhBiasParams& awhBiasParams,
                                      EnergyUnit           energyUnit,
                                      real                 kTValue);

    /*! \brief Opens a single output file for a bias, prints title and legends.
     *
     * \param[in] time  The time for of frame to be written.
     * \param[in] oenv  The output environment.
     * \returns the file pointer.
     */
    FILE* openBiasOutputFile(double time, const gmx_output_env_t* oenv) const;

    /*! \brief Writes data selected from \p block to file.
     *
     * \param[in] block          The energy block with the data to write.
     * \param[in] subBlockStart  The sub-block start index.
     * \param[in] fp             The file pointer.
     */
    void writeData(const t_enxblock& block, int subBlockStart, FILE* fp) const;

private:
    std::string baseFilename_;       /**< Base of the output file name. */
    std::string title_;              /**< Title for the graph. */
    int         numDim_;             /**< Number of dimensions. */
    int         firstGraphSubBlock_; /**< Index in the energy sub-blocks for the first graph. */
    int         numGraph_;           /**< Number of actual graphs. */
    bool        useKTForEnergy_;     /**< Whether to use kT instead of kJ/mol for energy. */
    std::vector<real> scaleFactor_; /**< Scale factors from energy file data to output for each of the numGraph quantities. */

    std::vector<std::string> legend_; /**< Legends for the output */
    std::string              xLabel_; /**< Label for the x-axis. */
    std::string              yLabel_; /**< Label for the y-axis. */
};

/*! \brief All meta-data that is shared for all output files for one bias */
struct BiasOutputSetup
{
    //! Constructor.
    BiasOutputSetup(int                         subblockStart,
                    std::unique_ptr<OutputFile> awhOutputFile,
                    std::unique_ptr<OutputFile> frictionOutputFile) :
        subblockStart_(subblockStart),
        awhOutputFile_(std::move(awhOutputFile)),
        frictionOutputFile_(std::move(frictionOutputFile))
    {
    }

    //! Return the AWH output file data.
    const OutputFile& awhOutputFile() const
    {
        GMX_RELEASE_ASSERT(awhOutputFile_ != nullptr,
                           "awhOutputFile() called without initialized AWH output file");

        return *awhOutputFile_;
    }

    //! Return the a pointer to the friction output file data, can return nullptr
    const OutputFile* frictionOutputFile() const { return frictionOutputFile_.get(); }

    //! Return the starting subblock.
    int subblockStart() const { return subblockStart_; }

private:
    const int                   subblockStart_; /**< The start index of the subblocks to read. */
    std::unique_ptr<OutputFile> awhOutputFile_; /**< The standard AWH output file data. */
    std::unique_ptr<OutputFile> frictionOutputFile_; /**< The friction/metric tensor output file data */
};

/*! \brief All options and meta-data needed for the AWH output */
class AwhReader
{
public:
    //! Constructor
    AwhReader(const AwhParams&  awhParams,
              int               numFileOptions,
              const t_filenm*   filenames,
              AwhGraphSelection awhGraphSelection,
              EnergyUnit        energyUnit,
              real              kT,
              const t_enxblock* block);

    //! Extract and print AWH data for an AWH block of one energy frame
    void processAwhFrame(const t_enxblock& block, double time, const gmx_output_env_t* oenv) const;

private:
    std::vector<BiasOutputSetup> biasOutputSetups_; /**< The output setups, one for each AWH bias. */
public:
    const real kT_; /**< kB*T in kJ/mol. */
};

/*! \brief Searches for frames in the energy file that should be processed */
class AwhFrameSearcher
{
public:
    /*! \brief Constructor
     *
     * \param[in] fp      Energy file to be processed.
     * \param[in] skip    Step between valid frames that will be considered for writing.
     */
    AwhFrameSearcher(ener_file_t fp, int skip);

    //! Destructor
    ~AwhFrameSearcher();

    //! Attempts to return next AWH frame from the energy file that statisfies all conditions
    t_enxframe* nextAwhFrame();

    /*! \brief Checks whether currentFrame_ or previousFrame_ should be dumped
     *
     * \param[in] dump    Timestep for dumping a frame.
     * \returns Pointer to the frame that should be dumped or nullptr if no suitable frame found.
     */
    t_enxframe* checkDump(double dump);

private:
    //! Check if currentFrame_ has AWH block of data
    bool checkAwhBlock();
    //! Check if currentFrame_ is allowed by the -skip option
    bool checkSkip();

    //! Energy file to process
    ener_file_t fp_;
    //! Number of frames to skip until the next one
    int skip_;
    //! Latest frame read from energy file
    t_enxframe* currentFrame_ = nullptr;
    //! Flag for prescence of the valid currentFrame_
    bool hasCurrentFrame_ = false;
    //! Previous frame read from energy file
    t_enxframe* previousFrame_ = nullptr;
    //! Flag for prescence of the valid previousFrame_
    bool hasPreviousFrame_ = false;
    //! Counter of frames with AWH data that have been read
    int awhFrameCounter_ = 0;
};

namespace
{

//! Enum for selecting output file type.
enum class OutputFileType
{
    Awh,     //!< AWH, all data except friction tensor.
    Friction //!< The full friction tensor.
};

/*! \brief The maximum number of observables per subblock, not including the full friction tensor.
 *
 * This number, as well as the legends in make_legend() should match
 * the output that mdrun writes. It would be better to define these
 * values in a single location.
 */
constexpr int maxAwhGraphs = 6;

/*! \brief Constructs a legend for a standard awh output file */
std::vector<std::string> makeLegend(const AwhBiasParams& awhBiasParams,
                                    OutputFileType       outputFileType,
                                    size_t               numLegend)
{
    const std::array<std::string, maxAwhGraphs> legendBase = { { "PMF",
                                                                 "Coord bias",
                                                                 "Coord distr",
                                                                 "Ref value distr",
                                                                 "Target ref value distr",
                                                                 "Friction metric" } };

    std::vector<std::string> legend;
    /* Give legends to dimensions higher than the first */
    for (int d = 1; d < awhBiasParams.ndim(); d++)
    {
        legend.push_back(gmx::formatString("dim%d", d));
    }

    switch (outputFileType)
    {
        case OutputFileType::Awh:
        {
            /* Add as many legends as possible from the "base" legend list */
            size_t legendBaseIndex = 0;
            while (legend.size() < numLegend && legendBaseIndex < legendBase.size())
            {
                legend.push_back(legendBase[legendBaseIndex]);
                legendBaseIndex++;
            }
        }
        break;
        case OutputFileType::Friction:
            for (int i0 = 0; i0 < awhBiasParams.ndim(); i0++)
            {
                for (int i1 = 0; i1 <= i0; i1++)
                {
                    legend.push_back(gmx::formatString("%d,%d", i0, i1));
                }
            }
            break;
    }

    GMX_RELEASE_ASSERT(legend.size() == numLegend,
                       "The number of legends requested for printing and the number generated "
                       "should be the same");

    return legend;
}

} // namespace

OutputFile::OutputFile(const std::string& filename, const std::string& baseTitle, int numBias, int biasIndex) :
    numDim_(0), firstGraphSubBlock_(0), numGraph_(0), useKTForEnergy_(false)

{
    baseFilename_ = filename.substr(0, filename.find('.'));
    title_        = baseTitle;
    if (numBias > 1)
    {
        baseFilename_ += gmx::formatString("%d", biasIndex + 1);
        title_ += gmx::formatString(" %d", biasIndex + 1);
    }
}

void OutputFile::initializeAwhOutputFile(int                  subblockStart,
                                         int                  numSubBlocks,
                                         const AwhBiasParams& awhBiasParams,
                                         AwhGraphSelection    graphSelection,
                                         EnergyUnit           energyUnit,
                                         real                 kTValue)
{
    /* The first subblock with actual graph y-values is index 1 + ndim */
    numDim_             = awhBiasParams.ndim();
    firstGraphSubBlock_ = subblockStart + 1 + numDim_;
    if (graphSelection == AwhGraphSelection::All)
    {
        /* There are one metadata and ndim coordinate blocks */
        numGraph_ = std::min(numSubBlocks - 1 - numDim_, maxAwhGraphs);
    }
    else
    {
        numGraph_ = 1;
    }
    useKTForEnergy_ = (energyUnit == EnergyUnit::KT);
    scaleFactor_.resize(numGraph_, 1);
    if (!useKTForEnergy_)
    {
        /* The first two graphs are in units of energy, multiply by kT */
        std::fill(scaleFactor_.begin(), scaleFactor_.begin() + std::min(2, numGraph_), kTValue);
    }
    int numLegend = numDim_ - 1 + numGraph_;
    legend_       = makeLegend(awhBiasParams, OutputFileType::Awh, numLegend);
    /* We could have both length and angle coordinates in a single bias */
    xLabel_ = "(nm, deg or lambda state)";
    yLabel_ = useKTForEnergy_ ? "(k\\sB\\NT)" : "(kJ/mol)";
    if (graphSelection == AwhGraphSelection::All)
    {
        yLabel_ += gmx::formatString(
                ", (nm\\S-%d\\N or rad\\S-%d\\N), (-)", awhBiasParams.ndim(), awhBiasParams.ndim());
    }
}

/*! \brief Initializes the output file setup for the fricion output (note that the filename is not set here). */
void OutputFile::initializeFrictionOutputFile(int                  subBlockStart,
                                              int                  numSubBlocks,
                                              const AwhBiasParams& awhBiasParams,
                                              EnergyUnit           energyUnit,
                                              real                 kTValue)
{
    /* The first subblock with actual graph y-values is index 1 + ndim */
    numDim_               = awhBiasParams.ndim();
    int numTensorElements = (numDim_ * (numDim_ + 1)) / 2;

    /* The friction tensor elements are always the last subblocks */
    if (numSubBlocks < 1 + numDim_ + maxAwhGraphs + numTensorElements)
    {
        gmx_fatal(FARGS,
                  "You requested friction tensor output, but the AWH data in the energy file does "
                  "not contain the friction tensor");
    }
    GMX_ASSERT(numSubBlocks == 1 + numDim_ + maxAwhGraphs + numTensorElements,
               "The number of sub-blocks per bias should be 1 + ndim + maxAwhGraphs + (ndim*(ndim "
               "+ 1))/2");

    firstGraphSubBlock_ = subBlockStart + numSubBlocks - numTensorElements;
    numGraph_           = numTensorElements;
    useKTForEnergy_     = (energyUnit == EnergyUnit::KT);
    scaleFactor_.resize(numGraph_, useKTForEnergy_ ? 1 : kTValue);
    int numLegend = numDim_ - 1 + numGraph_;
    legend_       = makeLegend(awhBiasParams, OutputFileType::Friction, numLegend);
    xLabel_       = "(nm, deg or lambda state)";
    if (useKTForEnergy_)
    {
        yLabel_ = "friction/k\\sB\\NT (nm\\S-2\\Nps, rad\\S-2\\Nps or ps)";
    }
    else
    {
        yLabel_ =
                "friction (kJ mol\\S-1\\Nnm\\S-2\\Nps, kJ mol\\S-1\\Nrad\\S-2\\Nps or kJ "
                "mol\\S-1\\Nps)";
    }
}

AwhReader::AwhReader(const AwhParams&  awhParams,
                     int               numFileOptions,
                     const t_filenm*   filenames,
                     AwhGraphSelection awhGraphSelection,
                     EnergyUnit        energyUnit,
                     real              kT,
                     const t_enxblock* block) :
    kT_(kT)
{
    GMX_RELEASE_ASSERT(block != nullptr, "NULL pointer passed to initAwhReader");

    bool outputFriction = opt2bSet("-fric", numFileOptions, filenames);

    /* The first subblock of each AWH block has metadata about
     * the number of subblocks belonging to that AWH block.
     * (block->nsub gives only the total number of subblocks and not how
     * they are distributed between the AWH biases).
     */

    /* Keep track of the first subblock of this AWH */
    int subblockStart = 0;
    for (int k = 0; k < awhParams.numBias(); k++)
    {
        const AwhBiasParams& awhBiasParams = awhParams.awhBiasParams()[k];

        int numSubBlocks = static_cast<int>(block->sub[subblockStart].fval[0]);

        std::unique_ptr<OutputFile> awhOutputFile(new OutputFile(
                opt2fn("-o", numFileOptions, filenames), "AWH", awhParams.numBias(), k));

        awhOutputFile->initializeAwhOutputFile(
                subblockStart, numSubBlocks, awhBiasParams, awhGraphSelection, energyUnit, kT);

        std::unique_ptr<OutputFile> frictionOutputFile;
        if (outputFriction)
        {
            frictionOutputFile = std::make_unique<OutputFile>(
                    opt2fn("-fric", numFileOptions, filenames), "Friction tensor", awhParams.numBias(), k);

            frictionOutputFile->initializeFrictionOutputFile(
                    subblockStart, numSubBlocks, awhBiasParams, energyUnit, kT);
        }

        biasOutputSetups_.emplace_back(BiasOutputSetup(
                subblockStart, std::move(awhOutputFile), std::move(frictionOutputFile)));

        subblockStart += numSubBlocks;
    }
}

FILE* OutputFile::openBiasOutputFile(double time, const gmx_output_env_t* oenv) const
{
    std::string filename = baseFilename_ + gmx::formatString("_t%g.xvg", time);

    FILE* fp = xvgropen(filename.c_str(), title_.c_str(), xLabel_, yLabel_, oenv);
    xvgrLegend(fp, legend_, oenv);

    return fp;
}

/*! \brief Prints data selected by \p outputFile from \p block to \p fp */
void OutputFile::writeData(const t_enxblock& block, int subBlockStart, FILE* fp) const
{
    int numPoints = block.sub[subBlockStart + 1].nr;
    for (int j = 0; j < numPoints; j++)
    {
        /* Print the coordinates for numDim dimensions */
        for (int d = 0; d < numDim_; d++)
        {
            fprintf(fp, "\t%8.4f", block.sub[subBlockStart + 1 + d].fval[j]);
        }

        /* Print numGraph observables */
        for (int i = 0; i < numGraph_; i++)
        {
            fprintf(fp, "\t%g", block.sub[firstGraphSubBlock_ + i].fval[j] * scaleFactor_[i]);
        }

        fprintf(fp, "\n");
    }
}

void AwhReader::processAwhFrame(const t_enxblock& block, double time, const gmx_output_env_t* oenv) const
{
    /* We look for AWH data every energy frame and count the no of AWH frames found. We only extract every 'skip' AWH frame. */

    for (const auto& setup : biasOutputSetups_)
    {
        const int subStart = setup.subblockStart();

        /* Each frame and AWH instance extracted generates one xvg file. */
        {
            const OutputFile& awhOutputFile = setup.awhOutputFile();

            FILE* fpAwh = awhOutputFile.openBiasOutputFile(time, oenv);

            /* Now do the actual printing. Metadata in first subblock is treated separately. */
            fprintf(fpAwh,
                    "# AWH metadata: target error = %.2f kT = %.2f kJ/mol\n",
                    block.sub[subStart].fval[1],
                    block.sub[subStart].fval[1] * kT_);

            fprintf(fpAwh, "# AWH metadata: log sample weight = %4.2f\n", block.sub[subStart].fval[2]);

            awhOutputFile.writeData(block, subStart, fpAwh);

            gmx_ffclose(fpAwh);
        }

        const OutputFile* frictionOutputFile = setup.frictionOutputFile();
        if (frictionOutputFile != nullptr)
        {
            FILE* fpFriction = frictionOutputFile->openBiasOutputFile(time, oenv);

            frictionOutputFile->writeData(block, subStart, fpFriction);

            gmx_ffclose(fpFriction);
        }
    }
}

AwhFrameSearcher::AwhFrameSearcher(ener_file_t fp, int skip) : fp_(fp), skip_(skip)
{
    snew(currentFrame_, 1);
    snew(previousFrame_, 1);
}

AwhFrameSearcher::~AwhFrameSearcher()
{
    sfree(currentFrame_);
    sfree(previousFrame_);
}

bool AwhFrameSearcher::checkAwhBlock()
{
    GMX_ASSERT(currentFrame_ != nullptr, "currentFrame_ should be valid");
    return find_block_id_enxframe(currentFrame_, enxAWH, nullptr) != nullptr;
}

bool AwhFrameSearcher::checkSkip()
{
    GMX_ASSERT(currentFrame_ != nullptr, "currentFrame_ should be valid");
    return (skip_ == 0) || (awhFrameCounter_ % skip_ == 0);
}

t_enxframe* AwhFrameSearcher::checkDump(double dump)
{
    // Return value
    t_enxframe* resultFrame = nullptr;

    if (hasCurrentFrame_)
    {
        // Case 1: currentFrame_ timestep is above dump and previousFrame_ is valid
        if (currentFrame_->t >= dump && hasPreviousFrame_)
        {
            // Decide which frame timestep is closer to dump
            if (std::abs(previousFrame_->t - dump) > std::abs(currentFrame_->t - dump))
            {
                resultFrame = currentFrame_;
            }
            else
            {
                resultFrame = previousFrame_;
            }
        }

        // Case 2: currentFrame_ timestep is above dump and previousFrame_ is not valid
        if (currentFrame_->t >= dump && !hasPreviousFrame_)
        {
            // Suggest to dump currentFrame_
            resultFrame = currentFrame_;
        }
    }
    else
    {
        // Case 3: currentFrame_ is not valid check for previousFrame_ and suggest to dump it
        if (hasPreviousFrame_)
        {
            resultFrame = previousFrame_;
        }
    }

    return resultFrame;
}

t_enxframe* AwhFrameSearcher::nextAwhFrame()
{
    // Return value
    t_enxframe* resultFrame = nullptr;

    // Swap current and previous frames
    std::swap(currentFrame_, previousFrame_);
    hasPreviousFrame_ = hasCurrentFrame_;
    hasCurrentFrame_  = false;

    // Initialize loop variable
    bool foundNextFrame = false;

    do
    {
        // If currentFrame_ exists
        if (do_enx(fp_, currentFrame_))
        {
            int withinTimeRange = check_times(currentFrame_->t);

            // Check if currentFrame_ contains AWH block of data and within time range
            if (checkAwhBlock() && (withinTimeRange == 0))
            {
                // Check for skip condition
                if (checkSkip())
                {

                    resultFrame      = currentFrame_;
                    hasCurrentFrame_ = true;
                    foundNextFrame   = true;
                }

                awhFrameCounter_++;
            }

            // If currentFrame_ timestep larger than -e option then stop search
            if (withinTimeRange > 0)
            {
                foundNextFrame = true;
            }
        }
        else
        {
            // We are at the end of file so should exit the loop and return nullptr
            foundNextFrame = true;
        }

    } while (!foundNextFrame);

    return resultFrame;
}

/*! \brief The main function for the AWH tool */
int gmx_awh(int argc, char* argv[])
{
    const char*     desc[]     = { "[THISMODULE] extracts AWH data from an energy file.",
                           "One or two files are written per AWH bias per time frame.",
                           "The bias index, if more than one, is appended to the file, as well as",
                           "the time of the frame. By default only the PMF is printed.",
                           "With [TT]-more[tt] the bias, target and coordinate distributions",
                           "are also printed.",
                           "With [TT]-more[tt] the bias, target and coordinate distributions",
                           "are also printed, as well as the metric sqrt(det(friction_tensor))",
                           "normalized such that the average is 1.",
                           "Option [TT]-fric[tt] prints all components of the friction tensor",
                           "to an additional set of files." };
    static gmx_bool moreGraphs = FALSE;
    static int      skip       = 0;
    static real     tDump      = -1;
    static bool     doDump     = false;
    static gmx_bool kTUnit     = FALSE;

    t_pargs pa[] = {
        { "-dump", FALSE, etTIME, { &tDump }, "Dump frame nearest specified time (%t)" },
        { "-skip", FALSE, etINT, { &skip }, "Skip number of frames between data points" },
        { "-more", FALSE, etBOOL, { &moreGraphs }, "Print more output" },
        { "-kt",
          FALSE,
          etBOOL,
          { &kTUnit },
          "Print free energy output in units of kT instead of kJ/mol" }
    };

    ener_file_t       fp;
    t_inputrec        ir;
    gmx_enxnm_t*      enm = nullptr;
    int               nre;
    gmx_output_env_t* oenv;

    t_filenm  fnm[] = { { efEDR, "-f", nullptr, ffREAD },
                       { efTPR, "-s", nullptr, ffREAD },
                       { efXVG, "-o", "awh", ffWRITE },
                       { efXVG, "-fric", "friction", ffOPTWR } };
    const int nfile = asize(fnm);
    if (!parse_common_args(&argc,
                           argv,
                           PCA_CAN_VIEW | PCA_CAN_BEGIN | PCA_CAN_END,
                           nfile,
                           fnm,
                           asize(pa),
                           pa,
                           asize(desc),
                           desc,
                           0,
                           nullptr,
                           &oenv))
    {
        return 0;
    }

    doDump = opt2parg_bSet("-dump", asize(pa), pa);

    fp = open_enx(ftp2fn(efEDR, nfile, fnm), "r");
    do_enxnms(fp, &nre, &enm);

    /* We just need the AWH parameters from inputrec. These are used to initialize
       the AWH reader when we have a frame to read later on. */
    gmx_mtop_t mtop;
    int        natoms;
    matrix     box;
    read_tpx(ftp2fn(efTPR, nfile, fnm), &ir, box, &natoms, nullptr, nullptr, &mtop);

    if (!ir.bDoAwh)
    {
        gmx_fatal(FARGS, "No AWH data in %s\n", opt2fn("-f", nfile, fnm));
    }

    std::unique_ptr<AwhReader> awhReader;
    auto                       awhFrameSearcher = std::make_unique<AwhFrameSearcher>(fp, skip);
    t_enxframe*                frame            = awhFrameSearcher->nextAwhFrame();
    t_enxblock*                block            = nullptr;

    while (frame != nullptr)
    {
        // Extract AWH block of data from the frame
        block = find_block_id_enxframe(frame, enxAWH, nullptr);

        /* Currently we have the number of subblocks per AWH stored
         * in the first subblock (we cannot get this directly from the tpr),
         * so we have to read an AWH block before making the legends.
         */

        if (awhReader == nullptr)
        {
            AwhGraphSelection awhGraphSelection =
                    (moreGraphs ? AwhGraphSelection::All : AwhGraphSelection::Pmf);
            EnergyUnit energyUnit = (kTUnit ? EnergyUnit::KT : EnergyUnit::KJPerMol);
            awhReader             = std::make_unique<AwhReader>(
                    *ir.awhParams, nfile, fnm, awhGraphSelection, energyUnit, gmx::c_boltz * ir.opts.ref_t[0], block);
        }

        if (doDump)
        {
            // Exit the loop if dumping timestep has been reached
            if (awhFrameSearcher->checkDump(tDump) != nullptr)
            {
                break;
            }
        }
        else
        {
            awhReader->processAwhFrame(*block, frame->t, oenv);
        }

        // Search for the next valid frame
        frame = awhFrameSearcher->nextAwhFrame();
    }

    // If we have frame to dump then write it
    if (doDump)
    {
        frame = awhFrameSearcher->checkDump(tDump);
        if (frame != nullptr)
        {
            fprintf(stderr,
                    "\nDumping frame at t= %g %s\n",
                    output_env_conv_time(oenv, frame->t),
                    output_env_get_time_unit(oenv).c_str());
            block = find_block_id_enxframe(frame, enxAWH, nullptr);
            awhReader->processAwhFrame(*block, frame->t, oenv);
        }
    }

    fprintf(stderr, "\n");

    close_enx(fp);

    return 0;
}
