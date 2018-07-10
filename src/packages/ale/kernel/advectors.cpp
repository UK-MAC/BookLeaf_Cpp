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
#include "packages/ale/kernel/advectors.h"

#include <cmath>
#include <algorithm>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
fluxElVl(
        int id1,
        int id2,
        int ilsize, // nel1
        int iasize, // nel2
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NCORN> cnbasis,
        ConstView<double, VarDim, NFACE> fcdbasis,
        ConstView<double, VarDim>        elvar,
        View<double, VarDim, NFACE>      fcflux)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    id1--;
    id2--;

    // Initialise
    for (int iel = 0; iel < iasize; iel++) {
        fcflux(iel, 0) = 0.;
        fcflux(iel, 1) = 0.;
        fcflux(iel, 2) = 0.;
        fcflux(iel, 3) = 0.;
    }

    // Construct flux
    for (int i1 = id1; i1 <= id2; i1++) {
        int const i2 = i1 + 2;
        for (int iel = 0; iel < ilsize; iel++) {
            int const iel2 = elel(iel, i2);
            int j2 = elfc(iel, i2);

            int j1 = (i2 + 1) % NCORN;
            double const r3 = cnbasis(iel, i2) + cnbasis(iel, j1);

            j1 = (j2 + 1) % NCORN;
            double const w5 = r3 + cnbasis(iel2, j2) + cnbasis(iel2, j1);

            int const iel1 = elel(iel, i1);
            j2 = elfc(iel, i1);
            j1 = i1 + 1;
            double const r4 = cnbasis(iel, i1) + cnbasis(iel, j1);
            j1 = (j2 + 1) % NCORN;
            double const w6 = r4 + cnbasis(iel1, j2) + cnbasis(iel1, j1);

            double const rv = elvar(iel);
            double r1 = fcdbasis(iel, i1);
            double r2 = fcdbasis(iel, i2);

            double const w1 = rv - elvar(iel2);
            double const w2 = elvar(iel1) - rv;
            double const w3 = std::fabs(w1);
            double const w4 = std::fabs(w2);
            double const w7 = std::copysign(1.0, w2);
            double const w8 = (w1*w6*w6+w2*w5*w5)/(w5*w6*(w5+w6));

            double tmp = std::fabs(w8);
                   tmp = std::min(tmp, w3/w5);
                   tmp = std::min(tmp, w4/w6);

            double grad = w7 * tmp;
            if (w1 * w2 <= 0.) grad = 0.;

            r1 *= rv + grad*(r3 - 0.5 * r1);
            r2 *= rv - grad*(r4 - 0.5 * r2);
            fcflux(iel, i1) = r1;
            fcflux(iel, i2) = r2;
        }
    }
}



void
fluxNdVl(
        int ilsize,
        int iasize,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NCORN> cnbasis,
        ConstView<double, VarDim, NCORN> cndbasis,
        ConstView<double, VarDim, NCORN> cnvar,
        View<double, VarDim, NCORN>      cnflux)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #define IX(i) ((i)-1)

    // Initialise
    for (int iel = 0; iel < iasize; iel++) {
        cnflux(iel, 0) = 0.;
        cnflux(iel, 1) = 0.;
        cnflux(iel, 2) = 0.;
        cnflux(iel, 3) = 0.;
    }

    // Construct flux
    // XXX(timrlaw): kept it 1-indexed because it was too confusing to switch
    //               over
    for (int ilfcl = 1; ilfcl <= 2; ilfcl++) {
        int const ilfcr = ilfcl + 2;
        for (int icn = 1; icn <= 2; icn++) {
            int const ilndl = ilfcl + icn - 1;
            int const ilndr = (ilfcr - icn + 1) % NCORN + 1;
            for (int iel = 1; iel <= ilsize; iel++) {
                double rd = 0.;

                int const iell = elel(IX(iel), IX(ilfcl)) + 1;
                int const ielr = elel(IX(iel), IX(ilfcr)) + 1;
                int const ifcl = elfc(IX(iel), IX(ilfcl)) + 1;
                int const ifcr = elfc(IX(iel), IX(ilfcr)) + 1;
                int const ii = ilfcl + 2 * (icn - 1);

                if (cndbasis(IX(iel), IX(ii)) > 0.) {
                    int const ilnndl = (ifcl + icn) % NCORN + 1;
                    assert(ilnndl >= 1);
                    int const ilnndr = (ifcl - icn + 1) % NCORN + 1;
                    assert(ilnndr >= 1);
                    double const rv = cnvar(IX(iel), IX(ilndl));
                    rd = cnbasis(IX(iel), IX(ilndl)) - 0.5 * cndbasis(IX(iel), IX(ii));

                    double const w5 = cnbasis(IX(iel), IX(ilndl)) + cnbasis(IX(iel), IX(ilndr));
                    double const w6 = cnbasis(IX(iell), IX(ilnndl)) + cnbasis(IX(iell), IX(ilnndr));
                    double const w1 = rv - cnvar(IX(iell), IX(ilnndl));
                    double const w2 = cnvar(IX(iel), IX(ilndr)) - rv;

                    double const w3 = std::fabs(w1);
                    double const w4 = std::fabs(w2);
                    double const w7 = std::copysign(1.0, w2);
                    double const w8 = (w2*w6*w6+w1*w5*w5)/(w5*w6*(w5+w6));

                    double tmp = std::fabs(w8);
                           tmp = std::min(tmp, w3/w6);
                           tmp = std::min(tmp, w4/w5);

                    double grad = w7 * tmp;
                    if (w1 * w2 <= 0.) grad = 0.;
                    rd = cndbasis(IX(iel), IX(ii)) * (rv + grad * rd);

                } else if (cndbasis(IX(iel), IX(ii)) < 0.) {
                    int const ilnndl = (ifcr + icn - 2) % NCORN + 1;
                    assert(ilnndl >= 1);

                    int ilnndr = (ifcr - icn - 1) % NCORN;
                    if (ilnndr < 0) ilnndr = NCORN + ilnndr;
                    ilnndr++;
                    assert(ilnndr >= 1);

                    double const rv = cnvar(IX(iel), IX(ilndr));
                    rd = cnbasis(IX(iel), IX(ilndr)) + 0.5 * cndbasis(IX(iel), IX(ii));

                    double const w5 = cnbasis(IX(iel), IX(ilndl)) + cnbasis(IX(iel), IX(ilndr));
                    double const w6 = cnbasis(IX(ielr), IX(ilnndl)) + cnbasis(IX(ielr), IX(ilnndr));
                    double const w1 = rv - cnvar(IX(iel), IX(ilndl));
                    double const w2 = cnvar(IX(ielr), IX(ilnndr)) - rv;

                    double const w3 = std::fabs(w1);
                    double const w4 = std::fabs(w2);
                    double const w7 = std::copysign(1.0, w2);
                    double const w8 = (w1*w6*w6+w2*w5*w5)/(w5*w6*(w5+w6));

                    double tmp = std::fabs(w8);
                           tmp = std::min(tmp, w3/w5);
                           tmp = std::min(tmp, w4/w6);

                    double grad = -w7 * tmp;
                    if (w1 * w2 <= 0.) grad = 0.;
                    rd = cndbasis(IX(iel), IX(ii)) * (rv + grad * rd);
                }

                cnflux(IX(iel), IX(ilndl)) -= rd;
                cnflux(IX(iel), IX(ilndr)) += rd;
            }
        }
    }

    #undef IX
}



void
updateEl(
        int id1,
        int id2,
        int ilsize, // nel
        int iasize, // nel1
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim>        elbase0,
        ConstView<double, VarDim>        elbase1,
        ConstView<double, VarDim>        cut,
        ConstView<double, VarDim, NFACE> fcflux,
        View<double, VarDim>             elflux,
        View<double, VarDim>             elvar)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Calculate total flux
    kernel::sumFlux(id1, id2, ilsize, iasize, elel, elfc, fcflux, elflux);

    // Update variable
    for (int iel = 0; iel < ilsize; iel++) {
        bool const cond = elbase1(iel) > cut(iel);
        elvar(iel) = cond ?
            (elvar(iel) * elbase0(iel) + elflux(iel)) / elbase1(iel) :
            elvar(iel);
    }
}



void
updateNd(
        int iusize, // nnd
        int icsize __attribute__((unused)), // nel1
        int insize, // nnd1
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim>           ndeln,
        ConstView<int, VarDim>           ndelf,
        ConstView<int, VarDim>           ndel,
        ConstView<double, VarDim>        ndbase0,
        ConstView<double, VarDim>        ndbase1,
        ConstView<double, VarDim>        cut,
        ConstView<unsigned char, VarDim> active,
        ConstView<double, VarDim, NCORN> cnflux,
        View<double, VarDim>             ndflux,
        View<double, VarDim>             ndvar)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Construct total flux
    for (int ind = 0; ind < insize; ind++) {
        ndflux(ind) = 0.;
    }

    for (int ind = 0; ind < insize; ind++) {
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

            ndflux(ind) += cnflux(iel, icn);
        }
    }

    // Update variable
    for (int ind = 0; ind < iusize; ind++) {
        bool const cond = active(ind) && (ndbase1(ind) > cut(ind));
        ndvar(ind) = cond ?
            (ndvar(ind) * ndbase0(ind) + ndflux(ind)) / ndbase1(ind) :
            ndvar(ind);
    }
}



void
sumFlux(
        int id1,
        int id2,
        int ilsize, // nel
        int iasize, // nel1
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NFACE> fcflux,
        View<double, VarDim>             elflux)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    id1--;
    id2--;

    for (int iel = 0; iel < iasize; iel++) {
        elflux(iel) = 0.;
    }

    for (int i1 = id1; i1 <= id2; i1++) {
        int const i2 = i1 + 2;
        for (int iel = 0; iel < ilsize; iel++) {
            int const iel1 = elel(iel, i1);
            int const iel2 = elel(iel, i2);
            int const j1   = elfc(iel, i1);
            int const j2   = elfc(iel, i2);

            assert(iel1 < iasize);
            assert(iel2 < iasize);

            double w1 = fcflux(iel1, j1);
            double w2 = fcflux(iel2, j2);

            w1 = iel1 == iel ? 0. : w1;
            w2 = iel2 == iel ? 0. : w2;

            elflux(iel) = elflux(iel) - fcflux(iel, i1) - fcflux(iel, i2) + w1 + w2;
        }
    }
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
