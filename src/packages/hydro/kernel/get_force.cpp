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
#include "packages/hydro/kernel/get_force.h"

#include <cassert>
#include <cmath>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
getForcePressure(
        ConstView<double, VarDim>   elpressure,
        ConstView<double, VarDim>   a1,
        ConstView<double, VarDim>   a3,
        ConstView<double, VarDim>   b1,
        ConstView<double, VarDim>   b3,
        View<double, VarDim, NCORN> cnfx,
        View<double, VarDim, NCORN> cnfy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, nel),
            BOOKLEAF_DEVICE_LAMBDA (int const iel)
    {
        double const w1 = elpressure(iel);

        cnfx(iel, 0) = w1 * (-b3(iel) + b1(iel));
        cnfx(iel, 1) = w1 * ( b3(iel) + b1(iel));
        cnfx(iel, 2) = w1 * ( b3(iel) - b1(iel));
        cnfx(iel, 3) = w1 * (-b3(iel) - b1(iel));

        cnfy(iel, 0) = w1 * ( a3(iel) - a1(iel));
        cnfy(iel, 1) = w1 * (-a3(iel) - a1(iel));
        cnfy(iel, 2) = w1 * (-a3(iel) + a1(iel));
        cnfy(iel, 3) = w1 * ( a3(iel) + a1(iel));
    });
}



void
getForceViscosity(
        ConstView<double, VarDim, NCORN> cnviscx,
        ConstView<double, VarDim, NCORN> cnviscy,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, nel),
            BOOKLEAF_DEVICE_LAMBDA (int const iel)
    {
        cnfx(iel, 0) += cnviscx(iel, 0);
        cnfx(iel, 1) += cnviscx(iel, 1);
        cnfx(iel, 2) += cnviscx(iel, 2);
        cnfx(iel, 3) += cnviscx(iel, 3);

        cnfy(iel, 0) += cnviscy(iel, 0);
        cnfy(iel, 1) += cnviscy(iel, 1);
        cnfy(iel, 2) += cnviscy(iel, 2);
        cnfy(iel, 3) += cnviscy(iel, 3);
    });
}



void
getForceSubzonalPressure(
        double const *pmeritreg,
        ConstView<int, VarDim>           elreg,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elcs2,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<double, VarDim, NCORN> spmass,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // XXX Missing code here that can't be merged

    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, nel),
            BOOKLEAF_DEVICE_LAMBDA (int const iel)
    {
        double lfx[NCORN][NCORN];
        double lfy[NCORN][NCORN];

        // Info
        // XXX(timrlaw): Why the abs here? elreg should always be positive
        int const ireg = std::abs((int) elreg(iel));
        double const w1 = pmeritreg[ireg];

        // Centroid
        double const x3 = 0.25 * (cnx(iel, 0) + cnx(iel, 1) + cnx(iel, 2) + cnx(iel, 3));
        double const y3 = 0.25 * (cny(iel, 0) + cny(iel, 1) + cny(iel, 2) + cny(iel, 3));

        // Initialise local force
        for (int icn = 0; icn < NCORN; icn++) {
            for (int jcn = 0; jcn < NCORN; jcn++) {
                lfx[icn][jcn] = 0.;
                lfy[icn][jcn] = 0.;
            }
        }

        // Loop over sub-elements
        for (int j1 = 0; j1 < NCORN; j1++) {

            // Construct sub-volumes
            double const x1 = cnx(iel, j1);
            double const y1 = cny(iel, j1);

            int j2 = (j1 + 1) % NCORN;
            double const x2 = 0.5 * (x1 + cnx(iel, j2));
            double const y2 = 0.5 * (y1 + cny(iel, j2));

            j2 = (j1 + 3) % NCORN;
            double const x4 = 0.5 * (x1 + cnx(iel, j2));
            double const y4 = 0.5 * (y1 + cny(iel, j2));

            // XXX Missing code here that can't be merged

            double const w3 = 0.25 * (-x1 + x2 + x3 - x4);
            double const w4 = 0.25 * (-x1 - x2 + x3 + x4);
            double const w5 = 0.25 * (-y1 + y2 + y3 - y4);
            double const w6 = 0.25 * (-y1 - y2 + y3 + y4);

            // Calculate change in pressure
            double w2 = 4.0 * (w3 * w6 - w4 * w5);
            w2 = spmass(iel, j1) / w2;
            w2 -= eldensity(iel);
            w2 = elcs2(iel) * w2;

            // Add to forces
            lfx[0][j1] = w2 * ( w5 - w6);
            lfx[1][j1] = w2 * ( w5 + w6);
            lfx[2][j1] = w2 * (-w5 + w6);
            lfx[3][j1] = w2 * (-w5 - w6);
            lfy[0][j1] = w2 * (-w3 + w4);
            lfy[1][j1] = w2 * (-w3 - w4);
            lfy[2][j1] = w2 * ( w3 - w4);
            lfy[3][j1] = w2 * ( w3 + w4);
        }

        // Distribute forces
        double w2 = 0.5  * (lfx[3][0] + lfx[1][3]);
        double w3 = 0.5  * (lfx[1][0] + lfx[3][1]);
        double w4 = 0.5  * (lfx[1][1] + lfx[3][2]);
        double w5 = 0.5  * (lfx[3][3] + lfx[1][2]);
        double w6 = 0.25 * (lfx[2][0] + lfx[2][1] + lfx[2][2] + lfx[2][3]);

        cnfx(iel, 0) += w1 * (lfx[0][0] + w2 + w3 + w6);
        cnfx(iel, 1) += w1 * (lfx[0][1] + w4 + w3 + w6);
        cnfx(iel, 2) += w1 * (lfx[0][2] + w4 + w5 + w6);
        cnfx(iel, 3) += w1 * (lfx[0][3] + w2 + w5 + w6);

        w2 = 0.5  * (lfy[3][0] + lfy[1][3]);
        w3 = 0.5  * (lfy[1][0] + lfy[3][1]);
        w4 = 0.5  * (lfy[1][1] + lfy[3][2]);
        w5 = 0.5  * (lfy[3][3] + lfy[1][2]);
        w6 = 0.25 * (lfy[2][0] + lfy[2][1] + lfy[2][2] + lfy[2][3]);

        cnfy(iel, 0) += w1 * (lfy[0][0] + w2 + w3 + w6);
        cnfy(iel, 1) += w1 * (lfy[0][1] + w4 + w3 + w6);
        cnfy(iel, 2) += w1 * (lfy[0][2] + w4 + w5 + w6);
        cnfy(iel, 3) += w1 * (lfy[0][3] + w2 + w5 + w6);
    });
}



void
getForceHourglass(
        double dt,
        double const *kappareg,
        ConstView<int, VarDim>           elreg,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elarea,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Hourglass restoring force
    double const w4 = 1.0 / dt;
    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, nel),
            BOOKLEAF_DEVICE_LAMBDA (int const iel)
    {
        double w2 = cnu(iel, 0) - cnu(iel, 1) + cnu(iel, 2) - cnu(iel, 3);
        double w3 = cnv(iel, 0) - cnv(iel, 1) + cnv(iel, 2) - cnv(iel, 3);

        int const ireg = elreg(iel);
        double const w1 = -kappareg[ireg] * eldensity(iel) * elarea(iel) * w4;
        w2 = w1 * w2;
        w3 = w1 * w3;

        cnfx(iel, 0) += w2;
        cnfx(iel, 1) -= w2;
        cnfx(iel, 2) += w2;
        cnfx(iel, 3) -= w2;

        cnfy(iel, 0) += w3;
        cnfy(iel, 1) -= w3;
        cnfy(iel, 2) += w3;
        cnfy(iel, 3) -= w3;
    });
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
