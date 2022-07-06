/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 1991- The GROMACS Authors
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

#ifndef GMX_FILEIO_TRXIO_H
#define GMX_FILEIO_TRXIO_H

#include "config.h"

#include <vector>

#include "gromacs/fileio/pdbio.h"
#include "gromacs/math/vectypes.h"

struct gmx_mtop_t;
struct gmx_output_env_t;
struct t_atoms;
struct t_fileio;
struct t_topology;
struct t_trxframe;
struct gmx_tng_trajectory;

namespace gmx
{
template<typename>
class ArrayRef;
}

#if GMX_USE_PLUGINS
using VmdPluginPointer = gmx_vmdplugin_t;
#else
using VmdPluginPointer = void*;
#endif

/*! \brief
 * Trajectory file status type.
 *
 * Controls fp and information about frame details.
 */
class TrajectoryIOStatus
{
public:
    TrajectoryIOStatus();
    ~TrajectoryIOStatus();
    TrajectoryIOStatus(t_fileio* fio);
    TrajectoryIOStatus(t_fileio*           fio,
                       gmx_tng_trajectory* tng,
                       std::vector<char>&& persistentLine,
                       VmdPluginPointer    plugin,
                       real                time,
                       size_t              natoms,
                       size_t              flags);
    TrajectoryIOStatus(TrajectoryIOStatus&& other) noexcept;
    TrajectoryIOStatus& operator=(TrajectoryIOStatus&& other) noexcept;
    //! Reset internal counter. Needed for pdb reading.
    void resetCounter();
    //! Currently active flags.
    size_t flags() const { return flags_; }
    //! Returns the number of frames read from the trajectory
    int numFramesRead() const;
    //! Returns true if I/O has printed to stderr for current frame.
    bool havePrintForFrame(const gmx_output_env_t* oenv) const;
    //! Write indexed trajectory file, see write_trxframe. \p gc may be nullptr.
    int writeIndexedTrxframe(const t_trxframe* fr, gmx::ArrayRef<const int> index, gmx_conect gc);
    /*! \brief
     * Write a single frame to trajectory file.
     *
     * Only entries for which the bool is true will be written,
     * except for step, time, lambda and/or box, which may not be
     * omitted for certain trajectory formats.
     * The precision for .xtc and .gro is fr->prec, when fr->bPrec=false,
     * the precision is set to 1000.
     *
     * \param[in] fr Trajectory frame to write.
     * \param[in] gc PDB connection information, can be nullptr.
     */
    int writeTrxframe(struct t_trxframe* fr, gmx_conect gc);
    /*! \brief Write indexed frame to trajectory file.
     *
     * \param [in] index Set of atoms to write, can be
     *             empty to write all atoms.
     * \param [in] atoms Atom information, can be nullptr
     *             when type doesn't need atoms.
     * \param [in] step Which step we are writing for.
     * \param [in] time Current simulation time.
     * \param [in] box Simulation box.
     * \param [in] coordinates Current coordinates to write.
     *             Size needs to match index.
     * \param [in] velocities Current velocities to write. Can be
     *             empty if no velocities to write, otherwise needs
     *             to match size of index.
     * \param [in] gc Connection information for pdb output.
     */
    int writeTrajectory(gmx::ArrayRef<const int> index,
                        const t_atoms*           atoms,
                        size_t                   step,
                        real                     time,
                        matrix                   box,
                        gmx::ArrayRef<gmx::RVec> coordinates,
                        gmx::ArrayRef<gmx::RVec> velocities,
                        gmx_conect               gc);

    //! Convenience wrapper for old coordinate format.
    int writeTrajectory(gmx::ArrayRef<const int> index,
                        const t_atoms*           atoms,
                        size_t                   step,
                        real                     time,
                        matrix                   box,
                        rvec*                    coordinates,
                        rvec*                    velocities,
                        gmx_conect               gc);
    /*! \brief Write a trxframe to the TNG file in status.
     *
     * This function is needed because both TrajectoryIOStatus and
     * gmx_tng_trajectory_t are encapsulated, so client trajectory-writing
     * code with a t_trxstatus can't just call the TNG writing
     * function. */
    void writeTngFrame(struct t_trxframe* fr);
    /*! \brief
     * Read next frame into \p fr.
     *
     * \param [in] oenv Output envrionment.
     * \param [in] fr Frame handle to read data into.
     * \returns True if file reading was successful.
     */
    bool readNextFrame(const gmx_output_env_t* oenv, struct t_trxframe* fr);
    //! Obtain file I/O information object.
    t_fileio* getFileIO();
    //! Return time of final frame, only works for TNG and XTC.
    float timeOfFinalFrame();
    //! Access to current frame number.
    int currentFrame() const { return currentFrame_; }
    //! Increment current frame.
    void increment() { currentFrame_++; }
    //! Access to number of atoms in frame.
    size_t numAtoms() const { return numAtoms_; }
    //! Access TNG pointer.
    gmx_tng_trajectory* tng() { return tng_; }
    //! Access persistent line.
    char* persistentLine() { return persistentLine_.data(); }

private:
    //! Flags for controlling how to read first/next frame.
    size_t flags_ = 0;
    //! Current frame being worked on.
    int currentFrame_ = -1;
    /*!\ brief
     * Time of first frame that has been read in.
     *
     * Needed for skipping frames with -dt option.
     */
    real initialTime_ = 0;
    //! Internal time of current frame.
    real frameTime_ = 0;
    //! Status of file I/O operations.
    t_fileio* fio_ = nullptr;
    //! TNG trajectory if loaded.
    gmx_tng_trajectory* tng_ = nullptr;
    //! Number of atoms in current frame.
    size_t numAtoms_ = 0;
    //! Persistent line for reading g96 trajectories. No comment on this ...
    std::vector<char> persistentLine_;
#if GMX_USE_PLUGINS
    //! Needed for loading files through VMD functions.
    gmx_vmdplugin_t* vmdplugin_ = nullptr;
#endif
};
/* I/O function types */

/************************************************
 *             Trajectory functions
 ************************************************/
//! Convert precision \p prec in 1/(nm) to number of decimal places
int prec2ndec(real prec);

/*! \brief Convert number of decimal places \p numDec to trajectory precision in
 * 1/(nm) */
real ndec2prec(int numDec);

void clear_trxframe(struct t_trxframe* fr, gmx_bool bFirst);
/* Set all content gmx_booleans to FALSE.
 * When bFirst = TRUE, set natoms=-1, all pointers to NULL
 *                     and all data to zero.
 */

void setTrxFramePbcType(struct t_trxframe* fr, PbcType pbcType);
/* Set the type of periodic boundary conditions, pbcType=PbcType::Unset is not set */

/*! \brief
 * Set up TNG writing to \p out.
 *
 * Sets up \p out for writing TNG. If \p in != NULL and contains a TNG trajectory
 * some data, e.g. molecule system, will be copied over from \p in to the return value.
 * If \p in == NULL a file name (infile) of a TNG file can be provided instead
 * and used for copying data to the return value.
 * If there is no TNG input \p natoms is used to create "implicit atoms" (no atom
 * or molecular data present). If \p natoms == -1 the number of atoms are
 * not known (or there is already a TNG molecule system to copy, in which case
 * natoms is not required anyhow). If an group of indexed atoms are written
 * \p natoms must be the length of \p index. \p index_group_name is the name of the
 * index group.
 *
 * \param[in] filename Name of new TNG file.
 * \param[in] filemode How to open the output file.
 * \param[in] in Input file pointer or null.
 * \param[in] infile Input file name or null.
 * \param[in] natoms Number of atoms to write.
 * \param[in] mtop Pointer to system topology or null.
 * \param[in] index Array of atom indices.
 * \param[in] index_group_name Name of the group of atom indices.
 * \returns Output TNG file status object.
 */
TrajectoryIOStatus trjtools_gmx_prepare_tng_writing(const char*              filename,
                                                    char                     filemode,
                                                    TrajectoryIOStatus*      in,
                                                    const char*              infile,
                                                    int                      natoms,
                                                    const gmx_mtop_t*        mtop,
                                                    gmx::ArrayRef<const int> index,
                                                    const char*              index_group_name);

//! Open a new trajectory file and return valid datastructure.
TrajectoryIOStatus openTrajectoryFile(const std::string& outputFileName, const char* filemode);

/*! \brief
 * Check whether expression (\p a - \p b) mod \p c is 0
 *
 * Uses different margin depending on whether we use gmx double or not.
 * \return true if (a-b) MOD c = 0
 */
bool bRmod(double a, double b, double c);

int check_times2(real t, real t0, bool bDouble);
/* This routine checkes if the read-in time is correct or not;
 * returns -1 if t<tbegin or t MOD dt = t0,
 *          0 if tbegin <= t <=tend+margin,
 *          1 if t>tend
 * where margin is 0.1*min(t-tp,tp-tpp), if this positive, 0 otherwise.
 * tp and tpp should be the time of the previous frame and the one before.
 * The mod is done with single or double precision accuracy depending
 * on the value of bDouble.
 */

int check_times(real t);
/* This routine checkes if the read-in time is correct or not;
 * returns -1 if t<tbegin,
 *          0 if tbegin <= t <=tend,
 *          1 if t>tend
 */


/* For trxframe.flags, used in trxframe read routines.
 * When a READ flag is set, the field will be read when present,
 * but a frame might be returned which does not contain the field.
 * When a NEED flag is set, frames not containing the field will be skipped.
 */
static constexpr size_t trxReadCoordinates = 1U << 0U;
static constexpr size_t trxNeedCoordinates = 1U << 1U;
static constexpr size_t trxReadVelocities  = 1U << 2U;
static constexpr size_t trxNeedVelocities  = 1U << 3U;
static constexpr size_t trxReadForces      = 1U << 4U;
static constexpr size_t trxNeedForces      = 1U << 5U;
/* Useful for reading natoms from a trajectory without skipping */
static constexpr size_t trxDontSkip = 1U << 6U;

/* For trxframe.not_ok */
static constexpr size_t trxframeHeaderNotOk = 1U << 0U;
static constexpr size_t trxframeDataNotOk   = 1U << 1U;

static constexpr size_t trxframeNotOk = trxframeHeaderNotOk | trxframeDataNotOk;

/*! \brief
 * Read first trajectory frame.
 *
 * Depending on \p flags, files are read in or reading is aborted.
 * Allocated memory for entries that should be read in.
 *
 * \returns New frame if reading succeeded.
 * \param [in] oenv Output environment controlling I/O
 * \param [in] fileName Name of trajectory file to read.
 * \param [inout] fr Handle to frame to read data into.
 * \param [in] flags Values to control what should be read in.
 */
std::optional<TrajectoryIOStatus> read_first_frame(const gmx_output_env_t* oenv,
                                                   const std::string&      fileName,
                                                   t_trxframe*             fr,
                                                   size_t                  flags);

struct t_topology* read_top(const char* fn, PbcType* pbcType);
/* Extract a topology data structure from a topology file.
 * If pbcType!=NULL *pbcType gives the pbc type.
 */

#endif
