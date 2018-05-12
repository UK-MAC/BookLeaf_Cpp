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



namespace bookleaf {
namespace hydro {
namespace kernel {
namespace {

inline double
denom(double x1, double y1, double x2, double y2)
{
    double w1 = y1 - y2;
    double w2 = x1 - x2;
    return w1*w1 + w2*w2;
}



inline double
distpp(double x3, double y3, double x4, double y4, double x1, double y1)
{
    double w1 = 0.5 * (x3 + x4) - x1;
    double w2 = 0.5 * (y3 + y4) - y1;
    return w1*w1 + w2*w2;
}



inline double
distpl(double x3, double y3, double x4, double y4, double x1, double y1,
        double x2, double y2)
{
    double w1 = 0.5 * (y1 - y2) * (x3 + x4) +
                0.5 * (y3 + y4) * (x2 - x1) + y2 * x1 - y1*x2;
    return w1*w1;
}



// These two kernels were originally located in the geometry utility
inline void
dlm(
        int iel,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
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



inline void
dln(
        double zcut,
        int iel,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
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
        unsigned char const *zdtnotreg,
        unsigned char const *zmidlength,
        ConstView<int, VarDim>           elreg,
        ConstView<double, VarDim>        elcs2,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, VarDim>             rscratch11,
        View<double, VarDim>             rscratch12,
        double &rdt,
        int &idt,
        std::string &sdt,
        Error &err)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Calculate CFL condition
    for (int iel = 0; iel < nel; iel++) {
        double result[NCORN];

        int ireg = elreg(iel);
        if (zdtnotreg[ireg]) {
            rscratch11(iel) = std::numeric_limits<double>::max();
            rscratch12(iel) = std::numeric_limits<double>::min();

        } else {
            if (zmidlength[ireg]) dlm(iel, cnx, cny, result);
            else                  dln(zcut, iel, cnx, cny, result);

            // Minimise result
            double w1 = result[0];
            for (int i = 1; i < NCORN; i++) {
                w1 = (result[i] < w1) ? result[i] : w1;
            }

            rscratch11(iel) = w1/elcs2(iel);
            rscratch12(iel) = w1;
        }
    }

    // Find minimum CFL condition
    int min_idx = 0;
    for (int iel = 1; iel < nel; iel++) {
        if (rscratch11(iel) < rscratch11(min_idx)) min_idx = iel;
    }

    double w1 = rscratch11(min_idx);
    if (w1 < 0) {
        FAIL_WITH_LINE(err, "ERROR");
        return;
    }

    rdt = cfl_sf*sqrt(w1);
    idt = min_idx;
    sdt = "     CFL";
}



void
getDtDiv(
        int nel,
        double div_sf,
        ConstView<double, VarDim>        a1,
        ConstView<double, VarDim>        a3,
        ConstView<double, VarDim>        b1,
        ConstView<double, VarDim>        b3,
        ConstView<double, VarDim>        elvolume,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    double w2 = std::numeric_limits<double>::min();
    int min_idx = 0;
    for (int iel = 0; iel < nel; iel++) {
        double w1 = cnu(iel, 0) * (-b3(iel) + b1(iel)) +
                    cnv(iel, 0) * ( a3(iel) - a1(iel)) +
                    cnu(iel, 1) * ( b3(iel) + b1(iel)) +
                    cnv(iel, 1) * (-a3(iel) - a1(iel)) +
                    cnu(iel, 2) * ( b3(iel) - b1(iel)) +
                    cnv(iel, 2) * (-a3(iel) + a1(iel)) +
                    cnu(iel, 3) * (-b3(iel) - b1(iel)) +
                    cnv(iel, 3) * ( a3(iel) + a1(iel));

        w1 = fabs(w1) / elvolume(iel);
        min_idx = w1 > w2 ? iel : min_idx;
        w2 = w1 > w2 ? w1 : w2;
    }

    rdt = div_sf/w2;
    idt = min_idx;
    sdt = "     DIV";
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
