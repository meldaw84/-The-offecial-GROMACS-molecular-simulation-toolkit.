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

#include <cmath>
#include <cstring>

#include <optional>

#include "gromacs/commandline/pargs.h"
#include "gromacs/fileio/confio.h"
#include "gromacs/fileio/trxio.h"
#include "gromacs/gmxana/gmx_ana.h"
#include "gromacs/gmxana/princ.h"
#include "gromacs/math/do_fit.h"
#include "gromacs/math/units.h"
#include "gromacs/math/utilities.h"
#include "gromacs/math/vec.h"
#include "gromacs/pbcutil/pbc.h"
#include "gromacs/pbcutil/rmpbc.h"
#include "gromacs/topology/index.h"
#include "gromacs/topology/topology.h"
#include "gromacs/trajectory/trajectoryframe.h"
#include "gromacs/utility/arrayref.h"
#include "gromacs/utility/arraysize.h"
#include "gromacs/utility/smalloc.h"

int gmx_filter(int argc, char* argv[])
{
    const char* desc[] = {
        "[THISMODULE] performs frequency filtering on a trajectory.",
        "The filter shape is cos([GRK]pi[grk] t/A) + 1 from -A to +A, where A is given",
        "by the option [TT]-nf[tt] times the time step in the input trajectory.",
        "This filter reduces fluctuations with period A by 85%, with period",
        "2*A by 50% and with period 3*A by 17% for low-pass filtering.",
        "Both a low-pass and high-pass filtered trajectory can be written.[PAR]",

        "Option [TT]-ol[tt] writes a low-pass filtered trajectory.",
        "A frame is written every [TT]-nf[tt] input frames.",
        "This ratio of filter length and output interval ensures a good",
        "suppression of aliasing of high-frequency motion, which is useful for",
        "making smooth movies. Also averages of properties which are linear",
        "in the coordinates are preserved, since all input frames are weighted",
        "equally in the output.",
        "When all frames are needed, use the [TT]-all[tt] option.[PAR]",

        "Option [TT]-oh[tt] writes a high-pass filtered trajectory.",
        "The high-pass filtered coordinates are added to the coordinates",
        "from the structure file. When using high-pass filtering use [TT]-fit[tt]",
        "or make sure you use a trajectory that has been fitted on",
        "the coordinates in the structure file."
    };

    static int      nf      = 10;
    static gmx_bool bNoJump = TRUE, bFit = FALSE, bLowAll = FALSE;
    t_pargs         pa[] = {
        { "-nf",
          FALSE,
          etINT,
          { &nf },
          "Sets the filter length as well as the output interval for low-pass filtering" },
        { "-all", FALSE, etBOOL, { &bLowAll }, "Write all low-pass filtered frames" },
        { "-nojump", FALSE, etBOOL, { &bNoJump }, "Remove jumps of atoms across the box" },
        { "-fit", FALSE, etBOOL, { &bFit }, "Fit all frames to a reference structure" }
    };
    const char *      topfile, *lowfile, *highfile;
    gmx_bool          bTop = FALSE;
    t_topology        top;
    PbcType           pbcType = PbcType::Unset;
    rvec*             xtop;
    matrix            topbox, *box, boxf;
    char*             grpname;
    int               isize;
    int*              index;
    real*             w_rls = nullptr;
    int               nffr, i, fr, j, d, m;
    int*              ind;
    real              flen, *filt, sum, *t;
    rvec              xcmtop, xcm, **x, *ptr, *xf, *xn, *xp, hbox;
    gmx_output_env_t* oenv;
    gmx_rmpbc_t       gpbc = nullptr;

    t_filenm fnm[] = { { efTRX, "-f", nullptr, ffREAD },
                       { efTPS, nullptr, nullptr, ffOPTRD },
                       { efNDX, nullptr, nullptr, ffOPTRD },
                       { efTRO, "-ol", "lowpass", ffOPTWR },
                       { efTRO, "-oh", "highpass", ffOPTWR } };
#define NFILE asize(fnm)

    if (!parse_common_args(
                &argc, argv, PCA_CAN_TIME | PCA_CAN_VIEW, NFILE, fnm, asize(pa), pa, asize(desc), desc, 0, nullptr, &oenv))
    {
        return 0;
    }

    highfile = opt2fn_null("-oh", NFILE, fnm);
    if (highfile)
    {
        topfile = ftp2fn(efTPS, NFILE, fnm);
        lowfile = opt2fn_null("-ol", NFILE, fnm);
    }
    else
    {
        topfile = ftp2fn_null(efTPS, NFILE, fnm);
        lowfile = opt2fn("-ol", NFILE, fnm);
    }
    if (topfile)
    {
        bTop = read_tps_conf(ftp2fn(efTPS, NFILE, fnm), &top, &pbcType, &xtop, nullptr, topbox, TRUE);
        if (bTop)
        {
            gpbc = gmx_rmpbc_init(&top.idef, pbcType, top.atoms.nr);
            gmx_rmpbc(gpbc, top.atoms.nr, topbox, xtop);
        }
    }

    clear_rvec(xcmtop);
    if (bFit)
    {
        fprintf(stderr, "Select group for least squares fit\n");
        get_index(&top.atoms, ftp2fn_null(efNDX, NFILE, fnm), 1, &isize, &index, &grpname);
        /* Set the weight */
        snew(w_rls, top.atoms.nr);
        for (i = 0; i < isize; i++)
        {
            w_rls[index[i]] = top.atoms.atom[index[i]].m;
        }
        calc_xcm(xtop, isize, index, top.atoms.atom, xcmtop, FALSE);
        for (j = 0; j < top.atoms.nr; j++)
        {
            rvec_dec(xtop[j], xcmtop);
        }
    }

    /* The actual filter length flen can actually be any real number */
    flen = 2 * nf;
    /* nffr is the number of frames that we filter over */
    nffr = 2 * nf - 1;
    snew(filt, nffr);
    sum = 0;
    for (i = 0; i < nffr; i++)
    {
        filt[i] = std::cos(2 * M_PI * (i - nf + 1) / static_cast<real>(flen)) + 1;
        sum += filt[i];
    }
    fprintf(stdout, "filter weights:");
    for (i = 0; i < nffr; i++)
    {
        filt[i] /= sum;
        fprintf(stdout, " %5.3f", filt[i]);
    }
    fprintf(stdout, "\n");

    snew(t, nffr);
    snew(x, nffr);
    snew(box, nffr);

    t_trxframe trxFr;
    auto status = read_first_frame(oenv, opt2fn("-f", NFILE, fnm), &trxFr, trxNeedCoordinates);
    snew(ind, trxFr.natoms);
    for (i = 0; i < trxFr.natoms; i++)
    {
        ind[i] = i;
    }
    /* x[nffr - 1] was already allocated by read_first_x */
    for (i = 0; i < nffr - 1; i++)
    {
        snew(x[i], trxFr.natoms);
    }
    snew(xf, trxFr.natoms);
    std::optional<TrajectoryIOStatus> outLow;
    std::optional<TrajectoryIOStatus> outHigh;
    if (lowfile)
    {
        outLow = openTrajectoryFile(lowfile, "w");
    }
    if (highfile)
    {
        outHigh = openTrajectoryFile(highfile, "w");
    }

    fr = 0;
    do
    {
        xn = trxFr.x;
        if (bNoJump && fr > 0)
        {
            xp = x[nffr - 2];
            for (j = 0; j < trxFr.natoms; j++)
            {
                for (d = 0; d < DIM; d++)
                {
                    hbox[d] = 0.5 * trxFr.box[d][d];
                }
            }
            for (i = 0; i < trxFr.natoms; i++)
            {
                for (m = DIM - 1; m >= 0; m--)
                {
                    if (hbox[m] > 0)
                    {
                        while (xn[i][m] - xp[i][m] <= -hbox[m])
                        {
                            for (d = 0; d <= m; d++)
                            {
                                xn[i][d] += trxFr.box[m][d];
                            }
                        }
                        while (xn[i][m] - xp[i][m] > hbox[m])
                        {
                            for (d = 0; d <= m; d++)
                            {
                                xn[i][d] -= trxFr.box[m][d];
                            }
                        }
                    }
                }
            }
        }
        if (bTop)
        {
            gmx_rmpbc(gpbc, trxFr.natoms, trxFr.box, xn);
        }
        if (bFit)
        {
            calc_xcm(xn, isize, index, top.atoms.atom, xcm, FALSE);
            for (j = 0; j < trxFr.natoms; j++)
            {
                rvec_dec(xn[j], xcm);
            }
            do_fit(trxFr.natoms, w_rls, xtop, xn);
            for (j = 0; j < trxFr.natoms; j++)
            {
                rvec_inc(xn[j], xcmtop);
            }
        }
        if (fr >= nffr && (outHigh.has_value() || bLowAll || fr % nf == nf - 1))
        {
            /* Lowpass filtering */
            for (j = 0; j < trxFr.natoms; j++)
            {
                clear_rvec(xf[j]);
            }
            clear_mat(boxf);
            for (i = 0; i < nffr; i++)
            {
                for (j = 0; j < trxFr.natoms; j++)
                {
                    for (d = 0; d < DIM; d++)
                    {
                        xf[j][d] += filt[i] * x[i][j][d];
                    }
                }
                for (j = 0; j < DIM; j++)
                {
                    for (d = 0; d < DIM; d++)
                    {
                        boxf[j][d] += filt[i] * box[i][j][d];
                    }
                }
            }
            if (outLow.has_value() && (bLowAll || fr % nf == nf - 1))
            {
                outLow->writeTrajectory(gmx::arrayRefFromArray(ind, trxFr.natoms),
                                        topfile ? &(top.atoms) : nullptr,
                                        0,
                                        trxFr.time,
                                        bFit ? topbox : boxf,
                                        xf,
                                        nullptr,
                                        nullptr);
            }
            if (outHigh.has_value())
            {
                /* Highpass filtering */
                for (j = 0; j < trxFr.natoms; j++)
                {
                    for (d = 0; d < DIM; d++)
                    {
                        xf[j][d] = xtop[j][d] + trxFr.x[j][d] - xf[j][d];
                    }
                }
                if (bFit)
                {
                    for (j = 0; j < trxFr.natoms; j++)
                    {
                        rvec_inc(xf[j], xcmtop);
                    }
                }
                for (j = 0; j < DIM; j++)
                {
                    for (d = 0; d < DIM; d++)
                    {
                        boxf[j][d] = topbox[j][d] + trxFr.box[j][d] - boxf[j][d];
                    }
                }
                outHigh->writeTrajectory(gmx::arrayRefFromArray(ind, trxFr.natoms),
                                         topfile ? &(top.atoms) : nullptr,
                                         0,
                                         trxFr.time,
                                         bFit ? topbox : boxf,
                                         xf,
                                         nullptr,
                                         nullptr);
            }
        }
        /* Cycle all the pointer and the box by one */
        ptr = x[0];
        for (i = 0; i < nffr - 1; i++)
        {
            t[i] = t[i + 1];
            x[i] = x[i + 1];
            copy_mat(box[i + 1], box[i]);
        }
        x[nffr - 1] = ptr;
        fr++;
    } while (status->readNextFrame(oenv, &trxFr));

    if (bTop)
    {
        gmx_rmpbc_done(gpbc);
    }

    return 0;
}
