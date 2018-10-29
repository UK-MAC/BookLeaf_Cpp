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
#include "packages/hydro/kernel/get_artificial_viscosity.h"

#include <cmath>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
initArtificialViscosity(
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim>             elvisc,
        DeviceView<double, VarDim, NFACE>      dx,
        DeviceView<double, VarDim, NFACE>      dy,
        DeviceView<double, VarDim, NFACE>      du,
        DeviceView<double, VarDim, NFACE>      dv,
        DeviceView<double, VarDim, NCORN>      cnviscx,
        DeviceView<double, VarDim, NCORN>      cnviscy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        elvisc(iel) = 0.;

        cnviscx(iel, 0) = 0.;
        cnviscx(iel, 1) = 0.;
        cnviscx(iel, 2) = 0.;
        cnviscx(iel, 3) = 0.;

        cnviscy(iel, 0) = 0.;
        cnviscy(iel, 1) = 0.;
        cnviscy(iel, 2) = 0.;
        cnviscy(iel, 3) = 0.;
    });

    // initialisation and gradient construction
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        for (int icn = 0; icn < NCORN; icn++) {
            int const icn2 = (icn + 1) % NCORN;
            du(iel, icn) = cnu(iel, icn2) - cnu(iel, icn);
            dv(iel, icn) = cnv(iel, icn2) - cnv(iel, icn);
            dx(iel, icn) = cnx(iel, icn2) - cnx(iel, icn);
            dy(iel, icn) = cny(iel, icn2) - cny(iel, icn);
        }
    });

    cudaSync();
}



void
limitArtificialViscosity(
        int nel,
        double zerocut,
        double cvisc1,
        double cvisc2,
        ConstDeviceView<int, VarDim>           ndtype,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim>        eldensity,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NFACE> du,
        ConstDeviceView<double, VarDim, NFACE> dv,
        ConstDeviceView<double, VarDim, NFACE> dx,
        ConstDeviceView<double, VarDim, NFACE> dy,
        DeviceView<double, VarDim, NCORN>      scratch,
        DeviceView<double, VarDim, NFACE>      edviscx,
        DeviceView<double, VarDim, NFACE>      edviscy,
        DeviceView<double, VarDim>             elvisc)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    for (int iside = 0; iside < NFACE / 2; iside++) {
        int const is1 = (iside + 3) % NFACE;
        int const is2 = iside + 1;

        Kokkos::parallel_for(
                RangePolicy(0, nel),
                KOKKOS_LAMBDA (int const iel)
        {
            // XXX(timrlaw): Manually fused these three loops
            {
                double tmp;

                // Connectivity
                int const in1 = elel(iel, iside);
                int const in2 = elel(iel, iside+2);

                // Edge 1
                double fw1 = du(iel, is1);
                double fw2 = dv(iel, is1);
                double fw3 = dx(iel, is1);
                double fw4 = dy(iel, is1);

                double den = BL_SQRT(fw1*fw1 + fw2*fw2);
                tmp = den > zerocut ? den : zerocut;
                den = 1.0 / tmp;
                double uhat = fw1 * den;
                double vhat = fw2 * den;

                den = BL_SQRT(fw3*fw3 + fw4*fw4);
                tmp = den > zerocut ? den : zerocut;
                den = 1.0 / tmp;
                double xhat = fw3 * den;
                double yhat = fw4 * den;

                den = fw3*xhat + fw4*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw1 = (fw1*uhat + fw2*vhat) / BL_SIGN(tmp, den);
                tmp = BL_FABS(fw1);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw1 = 1.0 / BL_SIGN(tmp, fw1);

                int ins = elfc(iel, iside);
                ins = (ins + 1) % NFACE;
                den = dx(in1, ins)*xhat + dy(in1, ins)*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw2 = (du(in1, ins)*uhat + dv(in1, ins)*vhat) / BL_SIGN(tmp, den);
                scratch(iel, 0) = fw2*fw1;

                ins = elfc(iel, iside+2);
                ins = (ins + 3) % NFACE;
                den = dx(in2, ins)*xhat + dy(in2, ins)*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw3 = (du(in2, ins)*uhat + dv(in2, ins)*vhat) / BL_SIGN(tmp, den);
                scratch(iel, 1) = fw3*fw1;

                // Edge 2
                fw1 = du(iel, is2);
                fw2 = dv(iel, is2);
                fw3 = dx(iel, is2);
                fw4 = dy(iel, is2);

                den = BL_SQRT(fw1*fw1 + fw2*fw2);
                tmp = den > zerocut ? den : zerocut;
                den = 1.0 / tmp;
                uhat = fw1 * den;
                vhat = fw2 * den;

                den = BL_SQRT(fw3*fw3 + fw4*fw4);
                tmp = den > zerocut ? den : zerocut;
                den = 1.0 / tmp;
                xhat = fw3 * den;
                yhat = fw4 * den;

                den = fw3*xhat + fw4*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw1 = (fw1*uhat + fw2*vhat) / BL_SIGN(tmp, den);
                tmp = BL_FABS(fw1);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw1 = 1.0 / BL_SIGN(tmp, fw1);

                ins = elfc(iel, iside);
                ins = (ins + 3) % NFACE;
                den = dx(in1, ins)*xhat + dy(in1, ins)*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw2 = (du(in1, ins)*uhat + dv(in1, ins)*vhat) / BL_SIGN(tmp, den);
                scratch(iel, 2) = fw2*fw1;

                ins = elfc(iel, iside+2);
                ins = (ins + 1) % NFACE;
                den = dx(in2, ins)*xhat + dy(in2, ins)*yhat;
                tmp = BL_FABS(den);
                tmp = tmp > zerocut ? tmp : zerocut;
                fw3 = (du(in2, ins)*uhat + dv(in2, ins)*vhat) / BL_SIGN(tmp, den);
                scratch(iel, 3) = fw3*fw1;
            }

            // BC
            int const bcins = iside + 2;
            {
                int const bcin1 = elel(iel, iside);
                int const bcin2 = elel(iel, bcins);

                bool in1 = bcin1 == iel;
                bool in2 = bcin2 == iel;

                int const ic11 = elnd(iel, iside);
                int const ic21 = elnd(iel, (iside + 1) % NFACE);
                bool cond1 = (ndtype(ic11) < 0) && (ndtype(ic21) < 0) && (bcin2 != iel);
                scratch(iel, 0) = in1 ? (cond1 ? 1.0 : 0.0) : scratch(iel, 0);
                scratch(iel, 2) = in1 ? (cond1 ? 1.0 : 0.0) : scratch(iel, 2);

                int const ic12 = elnd(iel, bcins);
                int const ic22 = elnd(iel, (bcins + 1) % NFACE);
                bool cond2 = (ndtype(ic12) < 0) && (ndtype(ic22) < 0) && (bcin1 != iel);
                scratch(iel, 1) = in2 ? (cond2 ? 1.0 : 0.0) : scratch(iel, 1);
                scratch(iel, 3) = in2 ? (cond2 ? 1.0 : 0.0) : scratch(iel, 3);
            }

            // Apply limiter
            {
                int limins = elel(iel, is1);

                double tmp;

                double w1 = cvisc1 * BL_SQRT(0.5 * (elcs2(iel) + elcs2(limins)));
                double w2 = scratch(iel, 0);
                double w3 = scratch(iel, 1);

                double const v1 = 0.5 * (w2 + w3);
                double const v2 = 2.0 * w2;
                double const v3 = 2.0 * w3;
                double v = v1 < v2 ? v1 : v2;
                       v = v < v3 ? v : v3;
                w2 = v < 1.0 ? v : 1.0;

                w2 = 0.0 > w2 ? 0.0 : w2;
                w3 = du(iel, is1);
                double w4 = dv(iel, is1);
                w4 = BL_SQRT(w3*w3 + w4*w4);
                w3 = 0.5 * (1.0 - w2) * (eldensity(iel) + eldensity(limins)) *
                    (w1 + cvisc2 * w4);

                edviscx(iel, is1) = w3;
                edviscy(iel, is1) = w3;

                tmp = eldensity(iel) > zerocut ? eldensity(iel) : zerocut;
                w4 = 2.0 * w4 / tmp;

                tmp = w3 * w4;
                elvisc(iel) = elvisc(iel) > tmp ? elvisc(iel) : tmp;

                limins = elel(iel, is2);
                w1 = cvisc1 * BL_SQRT(0.5 * (elcs2(iel) + elcs2(limins)));
                w2 = scratch(iel, 2);
                w3 = scratch(iel, 3);

                double const v4 = 0.5 * (w2 + w3);
                double const v5 = 2.0 * w2;
                double const v6 = 2.0 * w3;
                v = v4 < v5 ? v4 : v5;
                v = v < v6 ? v : v6;
                w2 = v < 1.0 ? v : 1.0;

                w2 = 0.0 > w2 ? 0.0 : w2;
                w3 = du(iel, is2);
                w4 = dv(iel, is2);
                w4 = BL_SQRT(w3*w3 + w4*w4);
                w3 = 0.5 * (1.0 - w2) * (eldensity(iel) + eldensity(limins)) *
                    (w1 + cvisc2 * w4);

                edviscx(iel, is2) = w3;
                edviscy(iel, is2) = w3;

                tmp = eldensity(iel) > zerocut ? eldensity(iel) : zerocut;
                w4 = 2.0 * w4 / tmp;

                tmp = w3 * w4;
                elvisc(iel) = elvisc(iel) > tmp ? elvisc(iel) : tmp;
            }
        });
    }

    cudaSync();
}



void
getArtificialViscosity(
        double zerocut,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim, NFACE>      cnviscx,
        DeviceView<double, VarDim, NFACE>      cnviscy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        double edviscx[NFACE];
        double edviscy[NFACE];

        double const cx = 0.25 * (cnx(iel,0)+cnx(iel,1)+cnx(iel,2)+cnx(iel,3));
        double const cy = 0.25 * (cny(iel,0)+cny(iel,1)+cny(iel,2)+cny(iel,3));

        for (int iside = 0; iside < NFACE; iside++) {
            int const ins = (iside + 1) % NFACE;

            double w1 = cnx(iel, iside);
            double w2 = cnx(iel, ins);
            double w3 = 0.5 * (w1 + w2);
            w1 = w2 - w1;
            w2 = cx;

            double w4 = cny(iel, iside);
            double w5 = cny(iel, ins);
            double w6 = 0.5 * (w4 + w5);
            w4 = w5 - w4;
            w5 = cy;

            double w7 = BL_SQRT((w2-w3)*(w2-w3) + (w5-w6)*(w5-w6));
            double w8 = BL_SQRT(w1*w1 + w4*w4);

            double den = 1.0 / w7;
            double xhat = (w5-w6) * den;
            double yhat = (w3-w2) * den;

            den = 1.0 / w8;
            w1 = w1 * den;
            w2 = w4 * den;
            w3 = xhat * w1 + yhat * w2;

            den = -BL_SIGN(1.0, w3) * w7;
            xhat = xhat * den;
            yhat = yhat * den;
            double uhat = cnu(iel, ins) - cnu(iel, iside);
            double vhat = cnv(iel, ins) - cnv(iel, iside);

            w5 = sqrt((uhat*uhat) + (vhat*vhat));
            w6 = uhat*xhat + vhat*yhat;

            den = w6 / (w5 > zerocut ? w5 : zerocut);

            edviscx[iside] = cnviscx(iel, iside) * uhat * den;
            edviscy[iside] = cnviscy(iel, iside) * vhat * den;

            // apply cut-off
            bool const cond = (w5 <= zerocut) ||
                              (w6 <= zerocut) ||
                              (w7 <= zerocut) ||
                              (w8 <= zerocut);

            edviscx[iside] = cond ? 0.0 : edviscx[iside];
            edviscy[iside] = cond ? 0.0 : edviscy[iside];
        }

        // convert from edge to corner
        cnviscx(iel, 0) = edviscx[0] - edviscx[3];
        cnviscx(iel, 1) = edviscx[1] - edviscx[0];
        cnviscx(iel, 2) = edviscx[2] - edviscx[1];
        cnviscx(iel, 3) = edviscx[3] - edviscx[2];
        cnviscy(iel, 0) = edviscy[0] - edviscy[3];
        cnviscy(iel, 1) = edviscy[1] - edviscy[0];
        cnviscy(iel, 2) = edviscy[2] - edviscy[1];
        cnviscy(iel, 3) = edviscy[3] - edviscy[2];
    });

    cudaSync();
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
