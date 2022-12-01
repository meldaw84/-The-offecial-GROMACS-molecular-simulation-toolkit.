/*
 * This file is part of the GROMACS molecular simulation package.
 *
 * Copyright 2012- The GROMACS Authors
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

#include "kernel_prune.h"

#include "gromacs/nbnxm/atomdata.h"
#include "gromacs/nbnxm/nbnxm_simd.h"
#include "gromacs/nbnxm/pairlist.h"
#include "gromacs/utility/gmxassert.h"

#ifdef GMX_NBNXN_SIMD_8XN
#    define GMX_SIMD_J_UNROLL_SIZE 1
#    include "kernel_common.h"
#endif

/* Prune a single nbnxn_pairtlist_t entry with distance rlistInner */
void nbnxn_kernel_prune_8xn(NbnxnPairlistCpu*              nbl,
                            const nbnxn_atomdata_t*        nbat,
                            gmx::ArrayRef<const gmx::RVec> shiftvec,
                            real                           rlistInner)
{
#ifdef GMX_NBNXN_SIMD_8XN
    using namespace gmx;

    /* We avoid push_back() for efficiency reasons and resize after filling */
    nbl->ci.resize(nbl->ciOuter.size());
    nbl->cj.resize(nbl->cjOuter.size());

    const nbnxn_ci_t* gmx_restrict ciOuter = nbl->ciOuter.data();
    nbnxn_ci_t* gmx_restrict       ciInner = nbl->ci.data();

    const nbnxn_cj_t* gmx_restrict cjOuter = nbl->cjOuter.data();
    nbnxn_cj_t* gmx_restrict       cjInner = nbl->cj.list_.data();

    const real* gmx_restrict x = nbat->x().data();

    const SimdReal rlist2_S(rlistInner * rlistInner);

    /* Initialize the new list count as empty and add pairs that are in range */
    int       nciInner = 0;
    int       ncjInner = 0;
    const int nciOuter = nbl->ciOuter.size();
    for (int ciIndex = 0; ciIndex < nciOuter; ciIndex++)
    {
        const nbnxn_ci_t* gmx_restrict ciEntry = &ciOuter[ciIndex];

        /* Copy the original list entry to the pruned entry */
        ciInner[nciInner].ci           = ciEntry->ci;
        ciInner[nciInner].shift        = ciEntry->shift;
        ciInner[nciInner].cj_ind_start = ncjInner;

        /* Extract shift data */
        int ish = (ciEntry->shift & NBNXN_CI_SHIFT);
        int ci  = ciEntry->ci;

        SimdReal shX_S = SimdReal(shiftvec[ish][XX]);
        SimdReal shY_S = SimdReal(shiftvec[ish][YY]);
        SimdReal shZ_S = SimdReal(shiftvec[ish][ZZ]);

#    if UNROLLJ <= 8
        int scix = ci * STRIDE * DIM;
#    else
        int scix = (ci >> 1) * STRIDE * DIM + (ci & 1) * (STRIDE >> 1);
#    endif

        /* Load i atom data */
        int      sciy  = scix + STRIDE;
        int      sciz  = sciy + STRIDE;
        std::array<std::array<SimdReal, DIM>, UNROLLI> xi;
        for (int i = 0; i < UNROLLI; i++)
        {
            xi[i][0] = SimdReal(x[scix + i]) + shX_S;
            xi[i][1] = SimdReal(x[sciy + i]) + shY_S;
            xi[i][2] = SimdReal(x[sciz + i]) + shZ_S;
        }

        for (int cjind = ciEntry->cj_ind_start; cjind < ciEntry->cj_ind_end; cjind++)
        {
            /* j-cluster index */
            int cj = cjOuter[cjind].cj;

            /* Atom indices (of the first atom in the cluster) */
#    if UNROLLJ == STRIDE
            int aj  = cj * UNROLLJ;
            int ajx = aj * DIM;
#    else
            int ajx = (cj >> 1) * DIM * STRIDE + (cj & 1) * UNROLLJ;
#    endif
            int ajy = ajx + STRIDE;
            int ajz = ajy + STRIDE;

            /* load j atom coordinates */
            SimdReal jx_S = load<SimdReal>(x + ajx);
            SimdReal jy_S = load<SimdReal>(x + ajy);
            SimdReal jz_S = load<SimdReal>(x + ajz);

            /* Calculate distance */
            std::array<std::array<SimdReal, DIM>, UNROLLI> d;
            for (int i = 0; i < UNROLLI; i++)
            {
                d[i][0] = xi[i][0] - jx_S;
                d[i][1] = xi[i][1] - jy_S;
                d[i][2] = xi[i][2] - jz_S;
            }

            /* rsq = dx*dx+dy*dy+dz*dz */
            std::array<SimdReal, UNROLLI> rsq;
            for (int i = 0; i < UNROLLI; i++)
            {
                rsq[i] = norm2(d[i][0], d[i][1], d[i][2]);
            }

            /* Do the cut-off check */
            std::array<SimdBool, UNROLLI> wco;
            for (int i = 0; i < UNROLLI; i++)
            {
                wco[i] = (rsq[i] < rlist2_S);
            }

            int offset = 1;
            while (offset < UNROLLI)
            {
                for (int i = 0; i < UNROLLI; i += 2 * offset)
                {
                    wco[i] = wco[i] || wco[i + offset];
                }
                offset *= 2;
            }

            /* Putting the assignment inside the conditional is slower */
            cjInner[ncjInner] = cjOuter[cjind];
            if (anyTrue(wco[0]))
            {
                ncjInner++;
            }
        }

        if (ncjInner > ciInner[nciInner].cj_ind_start)
        {
            ciInner[nciInner].cj_ind_end = ncjInner;
            nciInner++;
        }
    }

    nbl->ci.resize(nciInner);
    nbl->cj.resize(ncjInner);

#else /* GMX_NBNXN_SIMD_4XN */

    GMX_RELEASE_ASSERT(false, "4xN kernel called without 4xN support");

    GMX_UNUSED_VALUE(nbl);
    GMX_UNUSED_VALUE(nbat);
    GMX_UNUSED_VALUE(shiftvec);
    GMX_UNUSED_VALUE(rlistInner);

#endif /* GMX_NBNXN_SIMD_4XN */
}
