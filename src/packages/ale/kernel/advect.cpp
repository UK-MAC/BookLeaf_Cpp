/* @HEADER@
 * Crown Copyright 2018 AWE.
 *
 * This file is part of BookLeaf.
 *
 * BookLeaf is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * BookLeaf is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * BookLeaf. If not, see http://www.gnu.org/licenses/.
 * @HEADER@ */
#include "packages/ale/kernel/advect.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
updateBasisEl(
        double zerocut,
        double dencut,
        ConstView<double, VarDim> totv,
        ConstView<double, VarDim> totm,
        View<double, VarDim>      cutv,
        View<double, VarDim>      cutm,
        View<double, VarDim>      elvpr,
        View<double, VarDim>      elmpr,
        View<double, VarDim>      eldpr,
        View<double, VarDim>      elvolume,
        View<double, VarDim>      elmass,
        View<double, VarDim>      eldensity,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // update element basis
    #pragma omp parallel for
    for (int iel = 0; iel < nel; iel++) {

        // store basis variables
        elvpr(iel) = elvolume(iel);
        elmpr(iel) = elmass(iel);
        eldpr(iel) = eldensity(iel);

        // construct cut-off's
        cutv(iel) = zerocut;
        cutm(iel) = elvpr(iel) * dencut;

        // volume
        elvolume(iel) += totv(iel);

        // mass
        elmass(iel) += totm(iel);

        // density
        eldensity(iel) = elmass(iel) / elvolume(iel);
    }
}



void
initBasisNd(
        View<double, VarDim> ndv0,
        View<double, VarDim> ndv1,
        View<double, VarDim> ndm0,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // initialise basis
    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        ndv0(ind) = 0.;
        ndv1(ind) = 0.;
        ndm0(ind) = 0.;
    }
}



void
calcBasisNd(
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim>           ndeln,
        ConstView<int, VarDim>           ndelf,
        ConstView<int, VarDim>           ndel,
        ConstView<double, VarDim>        elv0,
        ConstView<double, VarDim>        elv1,
        ConstView<double, VarDim, NCORN> cnm1,
        View<double, VarDim>             ndv0,
        View<double, VarDim>             ndv1,
        View<double, VarDim>             ndm0,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        for (int i = 0; i < ndeln(ind); i++) {
            int const iel = ndel(ndelf(ind) + i);

            // Find element local corner number corresponding to ind
            int icn = -1;
            for (int jcn = 0; jcn < NCORN; jcn++) {
                if (elnd(iel, jcn) == ind) {
                    icn = jcn;
                    break;
                }
            }
            assert(icn >= 0 && "broken node-element mapping");

            double const w1 = 0.25 * elv0(iel);
            double const w2 = 0.25 * elv1(iel);
            double const w3 = cnm1(iel, icn);

            ndv0(ind) += w1;
            ndv1(ind) += w2;
            ndm0(ind) += w3;
        }
    }
}



void
fluxBasisNd(
        int id1,
        int id2,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<int, VarDim>           elsort __attribute__((unused)),
        ConstView<double, VarDim, NFACE> fcdv,
        ConstView<double, VarDim, NFACE> fcdm,
        View<double, VarDim, NCORN>      cndv,
        View<double, VarDim, NCORN>      cndm,
        View<double, VarDim, NCORN>      cnflux,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    id1--;
    id2--;

    #pragma omp parallel
    {

    // Initialise flux
    #pragma omp for
    for (int iel = 0; iel < nel; iel++) {
        cnflux(iel, 0) = 0.;
        cnflux(iel, 1) = 0.;
        cnflux(iel, 2) = 0.;
        cnflux(iel, 3) = 0.;
    }

    // Construct volume and mass flux
    for (int i1 = id1; i1 <= id2; i1++) {
        int const i2 = i1 + 2;

        // XXX(timrlaw): Not clear to me how the sort is having an effect here.
        //               cndv and cndm are set rather than accumulated, so
        //               ordering has no effect. cnflux is accumulated, but
        //               only once per iteration of the i1 loop per element, per
        //               corner, so changing the order doesn't matter here
        //               either.
        //for (int ii = 0; ii < nel; ii++) {
            //int const iel = elsort(ii);
        #pragma omp for
        for (int iel = 0; iel < nel; iel++) {

            int const ie1 = elel(iel, i1);
            int const ie2 = elel(iel, i2);
            int const is1 = elfc(iel, i1);
            int const is2 = elfc(iel, i2);

            double w1 = fcdv(ie1, is1);
            double w2 = fcdv(ie2, is2);
            double w3 = fcdm(ie1, is1);
            double w4 = fcdm(ie2, is2);

            w1 = ie1 == iel ? 0. : w1;
            w2 = ie2 == iel ? 0. : w2;
            w3 = ie1 == iel ? 0. : w3;
            w4 = ie2 == iel ? 0. : w4;

            w1 -= fcdv(iel, i1);
            w2 -= fcdv(iel, i2);
            w1 = 0.25 * (w1 - w2);
            cndv(iel, i1) = w1;
            cndv(iel, i2) = w1;

            w1 = w3 - fcdm(iel, i1);
            w2 = w4 - fcdm(iel, i2);
            w3 = 0.25 * (w1 - w2);
            cndm(iel, i1) = w3;
            cndm(iel, i2) = w3;

            w3 = 0.25 * (w1 + w2);
            cnflux(iel, 0) += w3;
            cnflux(iel, 1) += w3;
            cnflux(iel, 2) += w3;
            cnflux(iel, 3) += w3;
        }
    }

    } // #pragma omp parallel
}



void
massBasisNd(
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim>           ndeln,
        ConstView<int, VarDim>           ndelf,
        ConstView<int, VarDim>           ndel,
        ConstView<double, VarDim, NCORN> cnflux,
        View<double, VarDim, NCORN>      cnm1,
        View<double, VarDim>             ndm1,
        int nnd,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #pragma omp parallel
    {

    // Construct post nodal mass
    #pragma omp for
    for (int ind = 0; ind < nnd; ind++) {
        for (int i = 0; i < ndeln(ind); i++) {
            int const iel = ndel(ndelf(ind) + i);

            // Find element local corner number corresponding to ind
            int icn = -1;
            for (int jcn = 0; jcn < NCORN; jcn++) {
                if (elnd(iel, jcn) == ind) {
                    icn = jcn;
                    break;
                }
            }
            assert(icn >= 0 && "broken node-element mapping");

            ndm1(ind) += cnflux(iel, icn);
        }
    }

    // Construct post corner mass
    #pragma omp for
    for (int iel = 0; iel < nel; iel++) {
        for (int icn = 0; icn < NCORN; icn++) {
            cnm1(iel, icn) += cnflux(iel, icn);
        }
    }

    } // #pragma omp parallel
}



void
cutBasisNd(
        double cut,
        double dencut,
        ConstView<double, VarDim> ndv0,
        View<double, VarDim>      cutv,
        View<double, VarDim>      cutm,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Construct cut-offs
    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        cutv(ind) = cut;
        cutm(ind) = dencut * ndv0(ind);
    }
}



void
activeNd(
        int ibc,
        ConstView<int, VarDim>      ndstatus,
        ConstView<int, VarDim>      ndtype,
        View<unsigned char, VarDim> active,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Set active flag
    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        bool const is_active = (ndstatus(ind) > 0) &&
                               (ndtype(ind) != ibc) &&
                               (ndtype(ind) != -3);

        active(ind) = is_active ? 1 : 0;
    }
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
