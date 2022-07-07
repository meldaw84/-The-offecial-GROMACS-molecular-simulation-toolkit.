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
#include "gmxpre.h"

#include "gromacs/fileio/trxio.h"

#include "config.h"

#include <cassert>
#include <cmath>
#include <cstring>

#include <array>
#include <numeric>
#include <optional>
#include <vector>

#include "gromacs/fileio/checkpoint.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/filetypes.h"
#include "gromacs/fileio/g96io.h"
#include "gromacs/fileio/gmxfio.h"
#include "gromacs/fileio/gmxfio_xdr.h"
#include "gromacs/fileio/groio.h"
#include "gromacs/fileio/oenv.h"
#include "gromacs/fileio/pdbio.h"
#include "gromacs/fileio/timecontrol.h"
#include "gromacs/fileio/tngio.h"
#include "gromacs/fileio/tpxio.h"
#include "gromacs/fileio/trrio.h"
#include "gromacs/fileio/xdrf.h"
#include "gromacs/fileio/xtcio.h"
#include "gromacs/math/vec.h"
#include "gromacs/math/vectypes.h"
#include "gromacs/mdtypes/md_enums.h"
#include "gromacs/topology/atoms.h"
#include "gromacs/topology/symtab.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/exceptions.h"
#include "gromacs/utility/fatalerror.h"
#include "gromacs/utility/futil.h"
#include "gromacs/utility/gmxassert.h"
#include "gromacs/utility/smalloc.h"

#if GMX_USE_PLUGINS
#    include "gromacs/fileio/vmdio.h"
#endif

/* defines for frame counter output */
static constexpr int skip10   = 10;
static constexpr int skip100  = 100;
static constexpr int skip1000 = 1000;

/* utility functions */
template<bool haveDouble>
static bool bRmodInternal(double a, double b, double c)
{
    const double tol = 2 * (haveDouble ? GMX_DOUBLE_EPS : GMX_FLOAT_EPS);

    const int iq = static_cast<int>((a - b + tol * a) / c);

    return fabs(a - b - c * iq) <= tol * fabs(a);
}

bool bRmod(double a, double b, double c)
{
    return bRmodInternal<GMX_DOUBLE>(a, b, c);
}


int check_times2(real t, real t0, bool bDouble)
{
    int r;

#if !GMX_DOUBLE
    /* since t is float, we can not use double precision for bRmod */
    bDouble = false;
#endif

    r              = -1;
    auto startTime = timeValue(TimeControl::Begin);
    auto endTime   = timeValue(TimeControl::End);
    auto deltaTime = timeValue(TimeControl::Delta);
    if ((!startTime.has_value() || (t >= startTime.value()))
        && (!endTime.has_value() || (t <= endTime.value())))
    {
        bool isValidTime = false;
        if (bDouble)
        {
            isValidTime = deltaTime.has_value() && !bRmodInternal<true>(t, t0, deltaTime.value());
        }
        else
        {
            isValidTime = deltaTime.has_value() && !bRmodInternal<false>(t, t0, deltaTime.value());
        }

        if (isValidTime)
        {
            r = -1;
        }
        else
        {
            r = 0;
        }
    }
    else if (endTime.has_value() && (t >= endTime.value()))
    {
        r = 1;
    }
    if (debug)
    {
        fprintf(debug,
                "t=%g, t0=%g, b=%g, e=%g, dt=%g: r=%d\n",
                t,
                t0,
                startTime.value_or(0),
                endTime.value_or(0),
                deltaTime.value_or(0),
                r);
    }
    return r;
}

int check_times(real t)
{
    return check_times2(t, t, FALSE);
}

TrajectoryIOStatus::TrajectoryIOStatus(TrajectoryIOStatus&& other) noexcept :
    flags_(other.flags_),
    currentFrame_(other.currentFrame_),
    initialTime_(other.initialTime_),
    frameTime_(other.frameTime_),
    fio_(other.fio_),
    tng_(other.tng_),
    numAtoms_(other.numAtoms_),
    persistentLine_(std::move(other.persistentLine_))
{
#if GMX_USE_PLUGINS
    vmdplugin_       = other.vmdplugin_;
    other.vmdplugin_ = nullptr;
#endif
    other.fio_ = nullptr;
    other.tng_ = nullptr;
}

TrajectoryIOStatus& TrajectoryIOStatus::operator=(TrajectoryIOStatus&& other) noexcept
{
    flags_          = other.flags_;
    currentFrame_   = other.currentFrame_;
    initialTime_    = other.initialTime_;
    frameTime_      = other.frameTime_;
    fio_            = other.fio_;
    other.fio_      = nullptr;
    tng_            = other.tng_;
    other.tng_      = nullptr;
    numAtoms_       = other.numAtoms_;
    persistentLine_ = std::move(other.persistentLine_);
#if GMX_USE_PLUGINS
    vmdplugin_       = other.vmdplugin_;
    other.vmdplugin_ = nullptr;
#endif
    return *this;
}


TrajectoryIOStatus::TrajectoryIOStatus(t_fileio* fio) : fio_(fio) {}

TrajectoryIOStatus::TrajectoryIOStatus(t_fileio*           fio,
                                       gmx_tng_trajectory* tng,
                                       std::vector<char>&& persistentLine,
                                       VmdPluginPointer    plugin,
                                       real                time,
                                       size_t              natoms,
                                       size_t              flags) :
    flags_(flags),
    currentFrame_(0),
    initialTime_(time),
    frameTime_(time),
    fio_(fio),
    tng_(tng),
    numAtoms_(natoms),
    persistentLine_(persistentLine)
{
#if GMX_USE_PLUGINS
    vmdplugin_ = plugin;
#else
    GMX_UNUSED_VALUE(plugin);
#endif
}

void TrajectoryIOStatus::resetCounter()
{
    currentFrame_ = -1;
}

int TrajectoryIOStatus::numFramesRead() const
{
    return currentFrame_;
}

bool TrajectoryIOStatus::havePrintForFrame(const gmx_output_env_t* oenv) const
{
    return ((currentFrame_ < 2 * skip10 || currentFrame_ % skip10 == 0)
            && (currentFrame_ < 2 * skip100 || currentFrame_ % skip100 == 0)
            && (currentFrame_ < 2 * skip1000 || currentFrame_ % skip1000 == 0)
            && output_env_get_trajectory_io_verbosity(oenv) != 0);
}

static void printcountInternal(const TrajectoryIOStatus& status,
                               const gmx_output_env_t*   oenv,
                               const char*               l,
                               real                      t)
{
    if (status.havePrintForFrame(oenv))
    {
        fprintf(stderr, "\r%-14s %6d time %8.3f   ", l, status.currentFrame(), output_env_conv_time(oenv, t));
        fflush(stderr);
    }
}

static void printcount(const TrajectoryIOStatus& status, const gmx_output_env_t* oenv, real t, bool bSkip)
{
    printcountInternal(status, oenv, bSkip ? "Skipping frame" : "Reading frame", t);
}

static void printlast(const TrajectoryIOStatus& status, const gmx_output_env_t* oenv, real t)
{
    printcountInternal(status, oenv, "Last frame", t);
    fprintf(stderr, "\n");
    fflush(stderr);
}

static void printincomp(const TrajectoryIOStatus& status, t_trxframe* fr)
{
    if (fr->not_ok & trxframeHeaderNotOk)
    {
        fprintf(stderr, "WARNING: Incomplete header: nr %d time %g\n", status.currentFrame() + 1, fr->time);
    }
    else if (fr->not_ok)
    {
        fprintf(stderr, "WARNING: Incomplete frame: nr %d time %g\n", status.currentFrame() + 1, fr->time);
    }
    fflush(stderr);
}

static bool pdb_next_x(int currentFrame, FILE* fp, t_trxframe* fr)
{
    t_atoms   atoms;
    t_symtab* symtab;
    matrix    boxpdb;
    // Initiate model_nr to -1 rather than NOTSET.
    // It is not worthwhile introducing extra variables in the
    // read_pdbfile call to verify that a model_nr was read.
    PbcType pbcType;
    int     model_nr = -1, na;
    char    title[STRLEN], *time, *step;
    double  dbl;

    atoms.nr      = fr->natoms;
    atoms.atom    = nullptr;
    atoms.pdbinfo = nullptr;
    /* the other pointers in atoms should not be accessed if these are NULL */
    snew(symtab, 1);
    open_symtab(symtab);
    na = read_pdbfile(fp, title, &model_nr, &atoms, symtab, fr->x, &pbcType, boxpdb, nullptr);
    free_symtab(symtab);
    sfree(symtab);
    setTrxFramePbcType(fr, pbcType);
    if (currentFrame == 0)
    {
        fprintf(stderr, " '%s', %d atoms\n", title, fr->natoms);
    }
    fr->bPrec = TRUE;
    fr->prec  = 10000;
    fr->bX    = TRUE;
    fr->bBox  = (boxpdb[XX][XX] != 0.0);
    if (fr->bBox)
    {
        copy_mat(boxpdb, fr->box);
    }

    fr->step  = 0;
    step      = std::strstr(title, " step= ");
    fr->bStep = ((step != nullptr) && sscanf(step + 7, "%" SCNd64, &fr->step) == 1);

    dbl       = 0.0;
    time      = std::strstr(title, " t= ");
    fr->bTime = ((time != nullptr) && sscanf(time + 4, "%lf", &dbl) == 1);
    fr->time  = dbl;

    if (na == 0)
    {
        return FALSE;
    }
    else
    {
        if (na != fr->natoms)
        {
            gmx_fatal(FARGS, "Number of atoms in pdb frame %d is %d instead of %d", currentFrame, na, fr->natoms);
        }
        return TRUE;
    }
}

static int pdb_first_x(int currentFrame, FILE* fp, t_trxframe* fr)
{
    fprintf(stderr, "Reading frames from pdb file");
    frewind(fp);
    get_pdb_coordnum(fp, &fr->natoms);
    if (fr->natoms == 0)
    {
        gmx_fatal(FARGS, "\nNo coordinates in pdb file\n");
    }
    frewind(fp);
    snew(fr->x, fr->natoms);
    pdb_next_x(currentFrame, fp, fr);

    return fr->natoms;
}

std::optional<TrajectoryIOStatus>
read_first_frame(const gmx_output_env_t* oenv, const std::string& fn, t_trxframe* fr, size_t flags)
{
    t_fileio* fio = nullptr;
    bool      bOK;
    int       ftp = fn2ftp(fn.c_str());

    clear_trxframe(fr, TRUE);

    bool bFirst = true;

    std::optional<TrajectoryIOStatus> status;

    gmx_tng_trajectory* gmx_tng = nullptr;
    if (efTNG == ftp)
    {
        /* Special treatment for TNG files */
        gmx_tng_open(fn.c_str(), 'r', &gmx_tng);
    }
    else
    {
        fio = gmx_fio_open(fn.c_str(), "r");
    }
    std::vector<char> persistentLine;
    VmdPluginPointer  plugin        = nullptr;
    bool              printCount    = false;
    bool              printIncompat = false;
    switch (ftp)
    {
        case efTRR: break;
        case efCPT:
            read_checkpoint_trxframe(fio, fr);
            bFirst = false;
            break;
        case efG96:
        {

            /* Can not rewind a compressed file, so open it twice */
            persistentLine.resize(STRLEN + 1);
            t_symtab* symtab = nullptr;
            read_g96_conf(gmx_fio_getfp(fio), fn.c_str(), nullptr, fr, symtab, persistentLine.data());
            gmx_fio_close(fio);
            clear_trxframe(fr, false);
            if (flags & (trxReadCoordinates | trxNeedCoordinates))
            {
                snew(fr->x, fr->natoms);
            }
            if (flags & (trxReadVelocities | trxNeedVelocities))
            {
                snew(fr->v, fr->natoms);
            }
            fio = gmx_fio_open(fn.c_str(), "r");
            break;
        }
        case efXTC:
            if (read_first_xtc(fio, &fr->natoms, &fr->step, &fr->time, fr->box, &fr->x, &fr->prec, &bOK) == 0)
            {
                GMX_RELEASE_ASSERT(!bOK,
                                   "Inconsistent results - OK status from read_first_xtc, but 0 "
                                   "atom coords read");
                fr->not_ok = trxframeDataNotOk;
            }
            if (fr->not_ok)
            {
                fr->natoms    = 0;
                printIncompat = true;
            }
            else
            {
                fr->bPrec  = (fr->prec > 0);
                fr->bStep  = TRUE;
                fr->bTime  = TRUE;
                fr->bX     = TRUE;
                fr->bBox   = TRUE;
                printCount = true;
            }
            bFirst = false;
            break;
        case efTNG:
            fr->step = -1;
            if (!gmx_read_next_tng_frame(gmx_tng, fr, nullptr, 0))
            {
                fr->not_ok    = trxframeDataNotOk;
                fr->natoms    = 0;
                printIncompat = true;
            }
            else
            {
                printCount = true;
            }
            bFirst = false;
            break;
        case efPDB:
            pdb_first_x(0, gmx_fio_getfp(fio), fr);
            if (!fr->natoms)
            {
                printCount = true;
            }
            bFirst = false;
            break;
        case efGRO:
            if (gro_first_x_or_v(gmx_fio_getfp(fio), fr))
            {
                printCount = true;
            }
            bFirst = false;
            break;
        default:
#if GMX_USE_PLUGINS
            fprintf(stderr,
                    "The file format of %s is not a known trajectory format to GROMACS.\n"
                    "Please make sure that the file is a trajectory!\n"
                    "GROMACS will now assume it to be a trajectory and will try to open it using "
                    "the VMD plug-ins.\n"
                    "This will only work in case the VMD plugins are found and it is a trajectory "
                    "format supported by VMD.\n",
                    fn);
            gmx_fio_fp_close(fio); /*only close the file without removing FIO entry*/
            if (!read_first_vmd_frame(fn, &plugin, fr))
            {
                gmx_fatal(FARGS, "Not supported in read_first_frame: %s", fn.c_str());
            }
#else
            gmx_fatal(FARGS,
                      "Not supported in read_first_frame: %s. Please make sure that the file is a "
                      "trajectory.\n"
                      "GROMACS is not compiled with plug-in support. Thus it cannot read "
                      "non-GROMACS trajectory formats using the VMD plug-ins.\n"
                      "Please compile with plug-in support if you want to read non-GROMACS "
                      "trajectory formats.\n",
                      fn.c_str());
#endif
    }
    status.emplace(TrajectoryIOStatus{
            fio, gmx_tng, std::move(persistentLine), plugin, fr->time, static_cast<size_t>(fr->natoms), flags });
    if (printIncompat)
    {
        printincomp(status.value(), fr);
    }
    if (printCount)
    {
        status->increment();
        printcount(status.value(), oenv, fr->time, false);
    }

    /* Return FALSE if we read a frame that's past the set ending time. */
    if (!bFirst && (!(flags & trxDontSkip) && check_times(fr->time) > 0))
    {
        return std::nullopt;
    }

    if (bFirst || (!(flags & trxDontSkip) && check_times(fr->time) < 0))
    {
        /* Read a frame when no frame was read or the first was skipped */
        if (!status->readNextFrame(oenv, fr))
        {
            return std::nullopt;
        }
    }
    return fr->natoms > 0 ? std::move(status) : std::nullopt;
}


int prec2ndec(real prec)
{
    if (prec <= 0)
    {
        gmx_fatal(FARGS, "DEATH HORROR prec (%g) <= 0 in prec2ndec", prec);
    }

    return gmx::roundToInt(log(prec) / log(10.0));
}

real ndec2prec(int ndec)
{
    return pow(10.0, ndec);
}

t_fileio* TrajectoryIOStatus::getFileIO()
{
    return fio_;
}

float TrajectoryIOStatus::timeOfFinalFrame()
{
    int   filetype = gmx_fio_getftp(fio_);
    bool  bOK;
    float lasttime = -1;

    if (filetype == efXTC)
    {
        lasttime = xdr_xtc_get_last_frame_time(gmx_fio_getfp(fio_), gmx_fio_getxdr(fio_), numAtoms_, &bOK);
        if (!bOK)
        {
            gmx_fatal(FARGS, "Error reading last frame. Maybe seek not supported.");
        }
    }
    else if (filetype == efTNG)
    {
        if (!tng_)
        {
            gmx_fatal(FARGS, "Error opening TNG file.");
        }
        lasttime = gmx_tng_get_time_of_final_frame(tng_);
    }
    else
    {
        gmx_incons("Only supported for TNG and XTC");
    }
    return lasttime;
}

void clear_trxframe(t_trxframe* fr, gmx_bool bFirst)
{
    fr->not_ok    = 0;
    fr->bStep     = FALSE;
    fr->bTime     = FALSE;
    fr->bLambda   = FALSE;
    fr->bFepState = FALSE;
    fr->bAtoms    = FALSE;
    fr->bPrec     = FALSE;
    fr->bX        = FALSE;
    fr->bV        = FALSE;
    fr->bF        = FALSE;
    fr->bBox      = FALSE;
    if (bFirst)
    {
        fr->bDouble   = FALSE;
        fr->natoms    = -1;
        fr->step      = 0;
        fr->time      = 0;
        fr->lambda    = 0;
        fr->fep_state = 0;
        fr->atoms     = nullptr;
        fr->prec      = 0;
        fr->x         = nullptr;
        fr->v         = nullptr;
        fr->f         = nullptr;
        clear_mat(fr->box);
        fr->bPBC    = FALSE;
        fr->pbcType = PbcType::Unset;
        fr->bIndex  = false;
        fr->index   = nullptr;
    }
}

void setTrxFramePbcType(t_trxframe* fr, PbcType pbcType)
{
    fr->bPBC    = (pbcType == PbcType::Unset);
    fr->pbcType = pbcType;
}

int TrajectoryIOStatus::writeIndexedTrxframe(const t_trxframe* fr, gmx::ArrayRef<const int> index, gmx_conect gc)
{
    char title[STRLEN];

    std::vector<gmx::RVec>   xout;
    std::vector<gmx::RVec>   vout;
    std::vector<gmx::RVec>   fout;
    std::vector<int>         localIndex;
    gmx::ArrayRef<const int> indexView;
    if (index.empty())
    {
        localIndex.resize(fr->natoms);
        std::iota(localIndex.begin(), localIndex.end(), 0);
        indexView = localIndex;
    }
    else
    {
        indexView = index;
    }
    const real prec = fr->bPrec ? fr->prec : 1000.0;

    if (!tng_ && !fio_)
    {
        gmx_incons("No input file available");
    }
    const int ftp = tng_ ? efTNG : gmx_fio_getftp(fio_);

    switch (ftp)
    {
        case efTRR:
        case efTNG: break;
        default:
            if (!fr->bX)
            {
                gmx_fatal(FARGS, "Need coordinates to write a %s trajectory", ftp2ext(ftp));
            }
            break;
    }

    switch (ftp)
    {
        case efTRR:
        case efTNG:
            if (fr->bV)
            {
                vout.resize(indexView.size());
                for (size_t i = 0; i < indexView.size(); i++)
                {
                    copy_rvec(fr->v[indexView[i]], vout[i]);
                }
            }
            if (fr->bF)
            {
                fout.resize(indexView.size());
                for (size_t i = 0; i < indexView.size(); i++)
                {
                    copy_rvec(fr->f[indexView[i]], fout[i]);
                }
            }
            if (fr->bX)
            {
                xout.resize(indexView.size());
                for (size_t i = 0; i < indexView.size(); i++)
                {
                    copy_rvec(fr->x[indexView[i]], xout[i]);
                }
            }
            break;
        case efXTC:
            if (fr->bX)
            {
                xout.resize(indexView.size());
                for (size_t i = 0; i < indexView.size(); i++)
                {
                    copy_rvec(fr->x[indexView[i]], xout[i]);
                }
            }
            break;
        default: break;
    }

    switch (ftp)
    {
        case efTNG: gmx_write_tng_from_trxframe(tng_, fr, indexView.size()); break;
        case efXTC:
            write_xtc(fio_, indexView.size(), fr->step, fr->time, fr->box, as_rvec_array(xout.data()), prec);
            break;
        case efTRR:
            gmx_trr_write_frame(fio_,
                                numFramesRead(),
                                fr->time,
                                fr->step,
                                fr->box,
                                indexView.size(),
                                as_rvec_array(xout.data()),
                                as_rvec_array(vout.data()),
                                as_rvec_array(fout.data()));
            break;
        case efGRO:
        case efPDB:
        case efBRK:
        case efENT:
            if (!fr->bAtoms)
            {
                gmx_fatal(FARGS, "Can not write a %s file without atom names", ftp2ext(ftp));
            }
            sprintf(title, "frame t= %.3f", fr->time);
            if (ftp == efGRO)
            {
                write_hconf_indexed_p(gmx_fio_getfp(fio_),
                                      title,
                                      fr->atoms,
                                      indexView.size(),
                                      indexView.data(),
                                      fr->x,
                                      fr->bV ? fr->v : nullptr,
                                      fr->box);
            }
            else
            {
                write_pdbfile_indexed(gmx_fio_getfp(fio_),
                                      title,
                                      fr->atoms,
                                      fr->x,
                                      PbcType::Unset,
                                      fr->box,
                                      ' ',
                                      fr->step,
                                      indexView.size(),
                                      indexView.data(),
                                      gc,
                                      FALSE);
            }
            break;
        case efG96:
            sprintf(title, "frame t= %.3f", fr->time);
            write_g96_conf(gmx_fio_getfp(fio_), title, fr, indexView.size(), indexView.data());
            break;
        default: gmx_fatal(FARGS, "Sorry, write_trxframe_indexed can not write %s", ftp2ext(ftp));
    }

    return 0;
}

TrajectoryIOStatus trjtools_gmx_prepare_tng_writing(const char*              filename,
                                                    char                     filemode,
                                                    TrajectoryIOStatus*      in,
                                                    const char*              infile,
                                                    const int                natoms,
                                                    const gmx_mtop_t*        mtop,
                                                    gmx::ArrayRef<const int> index,
                                                    const char*              index_group_name)
{
    if (filemode != 'w' && filemode != 'a')
    {
        gmx_incons("Sorry, can only prepare for TNG output.");
    }

    gmx_tng_trajectory* inHandle  = (in != nullptr) ? in->tng() : nullptr;
    gmx_tng_trajectory* outHandle = nullptr;

    if (in != nullptr)
    {
        gmx_prepare_tng_writing(
                filename, filemode, &inHandle, &outHandle, natoms, mtop, index, index_group_name);
    }
    else if ((infile) && (efTNG == fn2ftp(infile)))
    {
        gmx_tng_trajectory_t tng_in;
        gmx_tng_open(infile, 'r', &tng_in);

        gmx_prepare_tng_writing(
                filename, filemode, &tng_in, &outHandle, natoms, mtop, index, index_group_name);
    }
    else
    {
        // we start from a file that is not a tng file or have been unable to load the
        // input file, so we need to populate the fields independently of it
        gmx_prepare_tng_writing(
                filename, filemode, nullptr, &outHandle, natoms, mtop, index, index_group_name);
    }
    return TrajectoryIOStatus{ nullptr, outHandle, {}, nullptr, 0, static_cast<size_t>(natoms), 0 };
}

void TrajectoryIOStatus::writeTngFrame(t_trxframe* frame)
{
    gmx_write_tng_from_trxframe(tng_, frame, frame->natoms);
}

int TrajectoryIOStatus::writeTrxframe(t_trxframe* fr, gmx_conect gc)
{
    char title[STRLEN];
    title[0]        = '\0';
    const real prec = fr->bPrec ? fr->prec : 1000.0;

    if (tng_ != nullptr)
    {
        gmx_tng_set_compression_precision(tng_, prec);
        writeTngFrame(fr);
        return 0;
    }

    switch (gmx_fio_getftp(fio_))
    {
        case efTRR: break;
        default:
            if (!fr->bX)
            {
                gmx_fatal(FARGS, "Need coordinates to write a %s trajectory", ftp2ext(gmx_fio_getftp(fio_)));
            }
            break;
    }

    switch (gmx_fio_getftp(fio_))
    {
        case efXTC: write_xtc(fio_, fr->natoms, fr->step, fr->time, fr->box, fr->x, prec); break;
        case efTRR:
            gmx_trr_write_frame(fio_,
                                fr->step,
                                fr->time,
                                fr->lambda,
                                fr->box,
                                fr->natoms,
                                fr->bX ? fr->x : nullptr,
                                fr->bV ? fr->v : nullptr,
                                fr->bF ? fr->f : nullptr);
            break;
        case efGRO:
        case efPDB:
        case efBRK:
        case efENT:
            if (!fr->bAtoms)
            {
                gmx_fatal(FARGS, "Can not write a %s file without atom names", ftp2ext(gmx_fio_getftp(fio_)));
            }
            sprintf(title, "frame t= %.3f", fr->time);
            if (gmx_fio_getftp(fio_) == efGRO)
            {
                write_hconf_p(
                        gmx_fio_getfp(fio_), title, fr->atoms, fr->x, fr->bV ? fr->v : nullptr, fr->box);
            }
            else
            {
                write_pdbfile(gmx_fio_getfp(fio_),
                              title,
                              fr->atoms,
                              fr->x,
                              fr->bPBC ? fr->pbcType : PbcType::Unset,
                              fr->box,
                              ' ',
                              fr->step,
                              gc);
            }
            break;
        case efG96: write_g96_conf(gmx_fio_getfp(fio_), title, fr, -1, nullptr); break;
        default:
            gmx_fatal(FARGS, "Sorry, write_trxframe can not write %s", ftp2ext(gmx_fio_getftp(fio_)));
    }

    return 0;
}

int TrajectoryIOStatus::writeTrajectory(gmx::ArrayRef<const int> index,
                                        const t_atoms*           atoms,
                                        size_t                   step,
                                        real                     time,
                                        matrix                   box,
                                        rvec*                    x,
                                        rvec*                    v,
                                        gmx_conect               gc)
{
    t_trxframe fr;

    clear_trxframe(&fr, TRUE);
    fr.bStep  = TRUE;
    fr.step   = step;
    fr.bTime  = TRUE;
    fr.time   = time;
    fr.bAtoms = atoms != nullptr;
    fr.atoms  = const_cast<t_atoms*>(atoms);
    fr.bX     = TRUE;
    fr.x      = x;
    fr.bV     = v != nullptr;
    fr.v      = v;
    fr.bBox   = TRUE;
    copy_mat(box, fr.box);

    return writeIndexedTrxframe(&fr, index, gc);
}

int TrajectoryIOStatus::writeTrajectory(gmx::ArrayRef<const int> index,
                                        const t_atoms*           atoms,
                                        size_t                   step,
                                        real                     time,
                                        matrix                   box,
                                        gmx::ArrayRef<gmx::RVec> x,
                                        gmx::ArrayRef<gmx::RVec> v,
                                        gmx_conect               gc)
{
    return writeTrajectory(
            index, atoms, step, time, box, as_rvec_array(x.data()), !v.empty() ? as_rvec_array(v.data()) : nullptr, gc);
}

TrajectoryIOStatus::~TrajectoryIOStatus()
{
    gmx_tng_close(&tng_);
    if (fio_)
    {
        gmx_fio_close(fio_);
        fio_ = nullptr;
    }
#if GMX_USE_PLUGINS
    sfree(vmdplugin_);
    vmdplugin_ = nullptr;
#endif
}

TrajectoryIOStatus openTrajectoryFile(const std::string& outfile, const char* filemode)
{
    if (filemode[0] != 'w' && filemode[0] != 'a' && filemode[1] != '+')
    {
        gmx_fatal(FARGS, "Sorry, write_trx can only write");
    }
    // special handling for TNG writing, in addition to dedicated function above.
    // this should remove a lot of the extra code paths that work around incompatibilities
    // of writing TNG compared to other file types.
    if (fn2ftp(outfile.c_str()) == efTNG)
    {
        return trjtools_gmx_prepare_tng_writing(
                outfile.c_str(), filemode[0], nullptr, nullptr, 0, nullptr, {}, nullptr);
    }
    else
    {
        return { gmx_fio_open(outfile.c_str(), filemode) };
    }
}

static bool gmx_next_frame(TrajectoryIOStatus* status, t_trxframe* fr)
{
    gmx_trr_header_t sh;
    gmx_bool         bOK, bRet;

    bRet = FALSE;

    if (gmx_trr_read_frame_header(status->getFileIO(), &sh, &bOK))
    {
        fr->bDouble   = sh.bDouble;
        fr->natoms    = sh.natoms;
        fr->bStep     = TRUE;
        fr->step      = sh.step;
        fr->bTime     = TRUE;
        fr->time      = sh.t;
        fr->bLambda   = TRUE;
        fr->bFepState = TRUE;
        fr->lambda    = sh.lambda;
        fr->bBox      = sh.box_size > 0;
        if (status->flags() & (trxReadCoordinates | trxNeedCoordinates))
        {
            if (fr->x == nullptr)
            {
                snew(fr->x, sh.natoms);
            }
            fr->bX = sh.x_size > 0;
        }
        if (status->flags() & (trxReadVelocities | trxNeedVelocities))
        {
            if (fr->v == nullptr)
            {
                snew(fr->v, sh.natoms);
            }
            fr->bV = sh.v_size > 0;
        }
        if (status->flags() & (trxReadForces | trxNeedForces))
        {
            if (fr->f == nullptr)
            {
                snew(fr->f, sh.natoms);
            }
            fr->bF = sh.f_size > 0;
        }
        if (gmx_trr_read_frame_data(status->getFileIO(), &sh, fr->box, fr->x, fr->v, fr->f))
        {
            bRet = TRUE;
        }
        else
        {
            fr->not_ok = trxframeDataNotOk;
        }
    }
    else if (!bOK)
    {
        fr->not_ok = trxframeHeaderNotOk;
    }

    return bRet;
}

bool TrajectoryIOStatus::readNextFrame(const gmx_output_env_t* oenv, t_trxframe* fr)
{
    int      ct;
    gmx_bool bOK, bMissingData = FALSE, bSkip = FALSE;
    bool     bRet = false;
    int      ftp;

    const real pt = frameTime_;

    do
    {
        clear_trxframe(fr, FALSE);

        if (tng_)
        {
            /* Special treatment for TNG files */
            ftp = efTNG;
        }
        else
        {
            ftp = gmx_fio_getftp(fio_);
        }
        auto startTime = timeValue(TimeControl::Begin);
        switch (ftp)
        {
            case efTRR: bRet = gmx_next_frame(this, fr); break;
            case efCPT:
                /* Checkpoint files can not contain mulitple frames */
                break;
            case efG96:
            {
                t_symtab* symtab = nullptr;
                read_g96_conf(gmx_fio_getfp(fio_), nullptr, nullptr, fr, symtab, persistentLine_.data());
                bRet = (fr->natoms > 0);
                break;
            }
            case efXTC:
                if (startTime.has_value() && (frameTime_ < startTime.value()))
                {
                    if (xtc_seek_time(fio_, startTime.value(), fr->natoms, TRUE))
                    {
                        gmx_fatal(FARGS,
                                  "Specified frame (time %f) doesn't exist or file "
                                  "corrupt/inconsistent.",
                                  startTime.value());
                    }
                    resetCounter();
                }
                bRet = (read_next_xtc(fio_, fr->natoms, &fr->step, &fr->time, fr->box, fr->x, &fr->prec, &bOK)
                        != 0);
                fr->bPrec = (bRet && fr->prec > 0);
                fr->bStep = bRet;
                fr->bTime = bRet;
                fr->bX    = bRet;
                fr->bBox  = bRet;
                if (!bOK)
                {
                    /* Actually the header could also be not ok,
                       but from bOK from read_next_xtc this can't be distinguished */
                    fr->not_ok = trxframeDataNotOk;
                }
                break;
            case efTNG: bRet = gmx_read_next_tng_frame(tng_, fr, nullptr, 0); break;
            case efPDB: bRet = pdb_next_x(currentFrame_, gmx_fio_getfp(fio_), fr); break;
            case efGRO: bRet = gro_next_x_or_v(gmx_fio_getfp(fio_), fr); break;
            default:
#if GMX_USE_PLUGINS
                bRet = read_next_vmd_frame(vmdplugin_, fr);
#else
                gmx_fatal(FARGS,
                          "DEATH HORROR in read_next_frame ftp=%s,status=%s",
                          ftp2ext(gmx_fio_getftp(fio_)),
                          gmx_fio_getname(fio_));
#endif
        }
        frameTime_ = fr->time;

        if (bRet)
        {
            bMissingData = ((((flags_ & trxNeedCoordinates) != 0) && !fr->bX)
                            || (((flags_ & trxNeedVelocities) != 0) && !fr->bV)
                            || (((flags_ & trxNeedForces) != 0) && !fr->bF));
            bSkip        = FALSE;
            if (!bMissingData)
            {
                ct = check_times2(fr->time, initialTime_, fr->bDouble);
                if (ct == 0 || ((flags_ & trxDontSkip) && ct < 0))
                {
                    increment();
                    printcount(*this, oenv, fr->time, FALSE);
                }
                else if (ct > 0)
                {
                    bRet = false;
                }
                else
                {
                    increment();
                    printcount(*this, oenv, fr->time, TRUE);
                    bSkip = TRUE;
                }
            }
        }

    } while (bRet && (bMissingData || bSkip));

    if (!bRet)
    {
        printlast(*this, oenv, pt);
        if (fr->not_ok)
        {
            printincomp(*this, fr);
        }
    }

    return bRet;
}

/***** T O P O L O G Y   S T U F F ******/

t_topology* read_top(const char* fn, PbcType* pbcType)
{
    int         natoms;
    PbcType     pbcTypeFile;
    t_topology* top;

    snew(top, 1);
    pbcTypeFile = read_tpx_top(fn, nullptr, nullptr, &natoms, nullptr, nullptr, top);
    if (pbcType)
    {
        *pbcType = pbcTypeFile;
    }

    return top;
}
