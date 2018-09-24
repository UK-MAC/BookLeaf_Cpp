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
#include "packages/hydro/kernel/get_acceleration.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
initAcceleration(
        View<double, VarDim> ndarea,
        View<double, VarDim> ndmass,
        View<double, VarDim> ndudot,
        View<double, VarDim> ndvdot,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        ndmass(ind) = 0.;
        ndarea(ind) = 0.;
        ndudot(ind) = 0.;
        ndvdot(ind) = 0.;
    }
}



void
scatterAcceleration(
        double zerocut,
        ConstView<int, VarDim>           ndeln,
        ConstView<int, VarDim>           ndelf,
        ConstView<int, VarDim>           ndel,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim, NCORN> cnwt,
        ConstView<double, VarDim, NCORN> cnmass,
        ConstView<double, VarDim, NCORN> cnfx,
        ConstView<double, VarDim, NCORN> cnfy,
        View<double, VarDim>             ndarea,
        View<double, VarDim>             ndmass,
        View<double, VarDim>             ndudot,
        View<double, VarDim>             ndvdot,
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

            double const density = eldensity(iel);
            double w = cnmass(iel, icn);
            w = w > zerocut ? w : cnmass(iel, (icn + (NCORN-1)) % NCORN);
            w = w > zerocut ? w : density * cnwt(iel, icn);

            ndmass(ind) += w;
            ndarea(ind) += cnwt(iel, icn);
            ndudot(ind) += cnfx(iel, icn);
            ndvdot(ind) += cnfy(iel, icn);
        }
    }
}



void
getAcceleration(
        double dencut,
        double zerocut,
        ConstView<double, VarDim> ndarea,
        View<double, VarDim>      ndmass,
        View<double, VarDim>      ndudot,
        View<double, VarDim>      ndvdot,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        double const w1 = dencut * ndarea(ind);

        double udot = ndmass(ind) > w1 ? ndudot(ind) / ndmass(ind) : 0.;
        double vdot = ndmass(ind) > w1 ? ndvdot(ind) / ndmass(ind) : 0.;
        double mass = ndmass(ind) > w1 ? ndmass(ind) : std::max(zerocut, w1);

        ndudot(ind) = udot;
        ndvdot(ind) = vdot;
        ndmass(ind) = mass;
    }
}



void
applyAcceleration(
        double dt,
        View<double, VarDim> ndubar,
        View<double, VarDim> ndvbar,
        View<double, VarDim> ndu,
        View<double, VarDim> ndv,
        int nnd)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    double const dt05 = 0.5 * dt;

    #pragma omp parallel for
    for (int ind = 0; ind < nnd; ind++) {
        double const w1 = ndu(ind);
        double const w2 = ndv(ind);

        ndu(ind) = w1 + dt*ndubar(ind);
        ndv(ind) = w2 + dt*ndvbar(ind);
        ndubar(ind) = w1 + dt05*ndubar(ind);
        ndvbar(ind) = w2 + dt05*ndvbar(ind);
    }
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
