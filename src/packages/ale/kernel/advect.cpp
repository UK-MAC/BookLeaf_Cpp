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
#include "common/cuda_utils.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
updateBasisEl(
        double zerocut,
        double dencut,
        ConstDeviceView<double, VarDim> totv,
        ConstDeviceView<double, VarDim> totm,
        DeviceView<double, VarDim>      cutv,
        DeviceView<double, VarDim>      cutm,
        DeviceView<double, VarDim>      elvpr,
        DeviceView<double, VarDim>      elmpr,
        DeviceView<double, VarDim>      eldpr,
        DeviceView<double, VarDim>      elvolume,
        DeviceView<double, VarDim>      elmass,
        DeviceView<double, VarDim>      eldensity,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // update element basis
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
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
    });

    cudaSync();
}



void
initBasisNd(
        DeviceView<double, VarDim> ndv0,
        DeviceView<double, VarDim> ndv1,
        DeviceView<double, VarDim> ndm0,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // initialise basis
    Kokkos::parallel_for(
            RangePolicy(0, nnd),
            KOKKOS_LAMBDA (int const ind)
    {
        ndv0(ind) = 0.;
        ndv1(ind) = 0.;
        ndm0(ind) = 0.;
    });

    cudaSync();
}



void
calcBasisNd(
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim>           ndeln,
        ConstDeviceView<int, VarDim>           ndelf,
        ConstDeviceView<int, VarDim>           ndel,
        ConstDeviceView<double, VarDim>        elv0,
        ConstDeviceView<double, VarDim>        elv1,
        ConstDeviceView<double, VarDim, NCORN> cnm1,
        DeviceView<double, VarDim>             ndv0,
        DeviceView<double, VarDim>             ndv1,
        DeviceView<double, VarDim>             ndm0,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    Kokkos::parallel_for(
            RangePolicy(0, nnd),
            KOKKOS_LAMBDA (int const ind)
    {
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
    });

    cudaSync();
}



void
fluxBasisNd(
        int id1,
        int id2,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<int, VarDim>           elsort __attribute__((unused)),
        ConstDeviceView<double, VarDim, NFACE> fcdv,
        ConstDeviceView<double, VarDim, NFACE> fcdm,
        DeviceView<double, VarDim, NCORN>      cndv,
        DeviceView<double, VarDim, NCORN>      cndm,
        DeviceView<double, VarDim, NCORN>      cnflux,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    id1--;
    id2--;

    // Initialise flux
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        cnflux(iel, 0) = 0.;
        cnflux(iel, 1) = 0.;
        cnflux(iel, 2) = 0.;
        cnflux(iel, 3) = 0.;
    });

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
        Kokkos::parallel_for(
                RangePolicy(0, nel),
                KOKKOS_LAMBDA (int const iel)
        {
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
        });
    }

    cudaSync();
}



void
massBasisNd(
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim>           ndeln,
        ConstDeviceView<int, VarDim>           ndelf,
        ConstDeviceView<int, VarDim>           ndel,
        ConstDeviceView<double, VarDim, NCORN> cnflux,
        DeviceView<double, VarDim, NCORN>      cnm1,
        DeviceView<double, VarDim>             ndm1,
        int nnd,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Construct post nodal mass
    Kokkos::parallel_for(
            RangePolicy(0, nnd),
            KOKKOS_LAMBDA (int const ind)
    {
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
    });

    // Construct post corner mass
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        for (int icn = 0; icn < NCORN; icn++) {
            cnm1(iel, icn) += cnflux(iel, icn);
        }
    });

    cudaSync();
}



void
cutBasisNd(
        double cut,
        double dencut,
        ConstDeviceView<double, VarDim> ndv0,
        DeviceView<double, VarDim>      cutv,
        DeviceView<double, VarDim>      cutm,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Construct cut-offs
    Kokkos::parallel_for(
            RangePolicy(0, nnd),
            KOKKOS_LAMBDA (int const ind)
    {
        cutv(ind) = cut;
        cutm(ind) = dencut * ndv0(ind);
    });

    cudaSync();
}



void
activeNd(
        int ibc,
        ConstDeviceView<int, VarDim>      ndstatus,
        ConstDeviceView<int, VarDim>      ndtype,
        DeviceView<unsigned char, VarDim> active,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Set active flag
    Kokkos::parallel_for(
            RangePolicy(0, nnd),
            KOKKOS_LAMBDA (int const ind)
    {
        bool const is_active = (ndstatus(ind) > 0) &&
                               (ndtype(ind) != ibc) &&
                               (ndtype(ind) != -3);

        active(ind) = is_active ? 1 : 0;
    });

    cudaSync();
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
