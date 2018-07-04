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
#include "packages/ale/kernel/get_dt.h"

#include <limits>
#include <cmath>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"
#include "common/reduce_idx.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
getDt(
        int nel,
        double zerocut,
        double ale_sf,
        bool zeul,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        ConstView<double, VarDim>        ellength,
        View<double, VarDim>             scratch,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

#ifdef BOOKLEAF_OPENMP_USER_REDUCTIONS
    ReduceIdx red { std::numeric_limits<double>::max(), -1 };

    if (zeul) {
        #pragma omp parallel for reduction(minloc:red)
        for (int iel = 0; iel < nel; iel++) {

            // Minimise node velocity squared
            double w1 = -std::numeric_limits<double>::max();
            for (int icn = 0; icn < NCORN; icn++) {
                double w2 = cnu(iel, icn) * cnu(iel, icn) +
                            cnv(iel, icn) * cnv(iel, icn);
                w1 = std::max(w1, w2);
            }

            w1 = ellength(iel) / std::max(w1, zerocut);
            ReduceIdx red2 { w1, iel };
            red = red2 < red ? red2 : red;
        }

    } else {
        // XXX Missing code that can't (or hasn't) been merged
    }

    rdt = ale_sf*sqrt(red.val);
    idt = red.idx;
    sdt = "     ALE";
#else
    double w2 = std::numeric_limits<double>::max();
    int minloc = nel;
    if (zeul) {
        #pragma omp parallel for reduction(min:w2)
        for (int iel = 0; iel < nel; iel++) {

            // Minimise node velocity squared
            double w1 = -std::numeric_limits<double>::max();
            for (int icn = 0; icn < NCORN; icn++) {
                double w2 = cnu(iel, icn) * cnu(iel, icn) +
                            cnv(iel, icn) * cnv(iel, icn);
                w1 = std::max(w1, w2);
            }

            w1 = ellength(iel) / std::max(w1, zerocut);

            scratch(iel) = w1;
            w2 = std::min(w2, w1);
        }

        #pragma omp parallel for reduction(min:minloc)
        for (int iel = 0; iel < nel; iel++) {
            double const w1 = scratch(iel);
            minloc = ((w1 == w2) && (minloc > iel)) ?
                iel :
                std::min(minloc, std::numeric_limits<int>::max());
        }

    } else {
        // XXX Missing code that can't (or hasn't) been merged
    }

    rdt = ale_sf*sqrt(w2);
    idt = minloc;
    sdt = "     ALE";
#endif
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
