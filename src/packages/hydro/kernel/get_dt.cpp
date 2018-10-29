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
#include "packages/hydro/kernel/get_dt.h"

#include <limits>
#include <cmath>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/error.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace hydro {
namespace kernel {
namespace {

KOKKOS_INLINE_FUNCTION double
denom(double x1, double y1, double x2, double y2)
{
    double w1 = y1 - y2;
    double w2 = x1 - x2;
    return w1*w1 + w2*w2;
}



KOKKOS_INLINE_FUNCTION double
distpp(double x3, double y3, double x4, double y4, double x1, double y1)
{
    double w1 = 0.5 * (x3 + x4) - x1;
    double w2 = 0.5 * (y3 + y4) - y1;
    return w1*w1 + w2*w2;
}



KOKKOS_INLINE_FUNCTION double
distpl(double x3, double y3, double x4, double y4, double x1, double y1,
        double x2, double y2)
{
    double w1 = 0.5 * (y1 - y2) * (x3 + x4) +
                0.5 * (y3 + y4) * (x2 - x1) + y2 * x1 - y1*x2;
    return w1*w1;
}



// These two kernels were originally located in the geometry utility
KOKKOS_INLINE_FUNCTION void
dlm(
        int iel,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        double res[NCORN])
{
    double elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    double x1 = elx[0] + elx[1];
    double x2 = elx[2] + elx[3];
    double y1 = ely[0] + ely[1];
    double y2 = ely[2] + ely[3];

    x1 = 0.5 * (x1 - x2);
    y1 = 0.5 * (y1 - y2);
    res[0] = x1*x1 + y1*y1;

    x1 = elx[2] + elx[1];
    x2 = elx[0] + elx[3];
    y1 = ely[2] + ely[1];
    y2 = ely[0] + ely[3];

    x1 = 0.5 * (x1 - x2);
    y1 = 0.5 * (y1 - y2);

    res[1] = x1*x1 + y1*y1;
    res[2] = res[0];
    res[3] = res[1];
}



KOKKOS_INLINE_FUNCTION void
dln(
        double zcut,
        int iel,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        double res[NCORN])
{
    double elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    for (int i = 0; i < NCORN; i++) {
        int const j1 = i;
        int const j2 = (i + 1) % NCORN;
        int const j3 = (i + 2) % NCORN;
        int const j4 = (i + 3) % NCORN;

        double const w1 = denom(elx[j3], ely[j3], elx[j4], ely[j4]);

        double const w2 = w1 < zcut ?
            distpp(elx[j1], ely[j1], elx[j2],
                   ely[j2], elx[j3], ely[j3]) :
            distpl(elx[j1], ely[j1], elx[j2],
                   ely[j2], elx[j3], ely[j3],
                   elx[j4], ely[j4]) / w1;

        res[i] = w2;
    }
}

} // namespace

void
getDtCfl(
        int nel,
        double zcut,
        double cfl_sf,
        ConstDeviceView<unsigned char, VarDim> zdtnotreg,
        ConstDeviceView<unsigned char, VarDim> zmidlength,
        ConstDeviceView<int, VarDim>           elreg,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        DeviceView<double, VarDim>             rscratch11,
        DeviceView<double, VarDim>             rscratch12,
        double &rdt,
        int &idt,
        std::string &sdt,
        Error &err)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Calculate CFL condition
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        int const ireg = elreg(iel);

        double resultm[NCORN];
        dlm(iel, cnx, cny, resultm);

        double resultn[NCORN];
        dln(zcut, iel, cnx, cny, resultn);

        double *result = zmidlength(ireg) ? resultm : resultn;

        // Minimise result
        double w1 = result[0];
        for (int i = 1; i < NCORN; i++) {
            w1 = (result[i] < w1) ? result[i] : w1;
        }

        rscratch11(iel) = w1/elcs2(iel);
        rscratch12(iel) = w1;

        rscratch11(iel) = zdtnotreg(ireg) ?
                            NPP_MAXABS_64F : rscratch11(iel);
        rscratch12(iel) = zdtnotreg(ireg) ?
                            NPP_MINABS_64F : rscratch12(iel);
    });

    // Find minimum CFL condition
    using MinLoc     = Kokkos::Experimental::MinLoc<double, int>;
    using MinLocType = MinLoc::value_type;

    MinLocType minloc;
    Kokkos::parallel_reduce(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel, MinLocType &lminloc)
    {
        double const w = rscratch11(iel);
        lminloc.loc = w < lminloc.val ? iel : lminloc.loc;
        lminloc.val = w < lminloc.val ? w   : lminloc.val;
    }, MinLoc::reducer(minloc));

    cudaSync();

    double w1 = minloc.val;
    if (w1 < 0) {
        FAIL_WITH_LINE(err, "ERROR");
        return;
    }

    rdt = cfl_sf*sqrt(w1);
    idt = minloc.loc;
    sdt = "     CFL";
}



void
getDtDiv(
        int nel,
        double div_sf,
        ConstDeviceView<double, VarDim>        a1,
        ConstDeviceView<double, VarDim>        a3,
        ConstDeviceView<double, VarDim>        b1,
        ConstDeviceView<double, VarDim>        b3,
        ConstDeviceView<double, VarDim>        elvolume,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    using MaxLoc     = Kokkos::Experimental::MaxLoc<double, int>;
    using MaxLocType = MaxLoc::value_type;

    MaxLocType maxloc;
    Kokkos::parallel_reduce(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel, MaxLocType &lmaxloc)
    {
        double w1 = cnu(iel, 0) * (-b3(iel) + b1(iel)) +
                    cnv(iel, 0) * ( a3(iel) - a1(iel)) +
                    cnu(iel, 1) * ( b3(iel) + b1(iel)) +
                    cnv(iel, 1) * (-a3(iel) - a1(iel)) +
                    cnu(iel, 2) * ( b3(iel) - b1(iel)) +
                    cnv(iel, 2) * (-a3(iel) + a1(iel)) +
                    cnu(iel, 3) * (-b3(iel) - b1(iel)) +
                    cnv(iel, 3) * ( a3(iel) + a1(iel));

        w1 = fabs(w1) / elvolume(iel);
        lmaxloc.loc = w1 > lmaxloc.val ? iel : lmaxloc.loc;
        lmaxloc.val = w1 > lmaxloc.val ? w1 : lmaxloc.val;
    }, MaxLoc::reducer(maxloc));

    cudaSync();

    rdt = div_sf/maxloc.val;
    idt = maxloc.loc;
    sdt = "     DIV";
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
