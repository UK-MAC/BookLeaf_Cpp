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
#include "common/cuda_utils.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
getDt(
        int nel,
        double zerocut,
        double ale_sf,
        bool zeul,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        ConstDeviceView<double, VarDim>        ellength,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    using MinLoc     = Kokkos::Experimental::MinLoc<double, int>;
    using MinLocType = MinLoc::value_type;

    MinLocType minloc;

    if (zeul) {
        Kokkos::parallel_reduce(
                RangePolicy(0, nel),
                KOKKOS_LAMBDA (int const iel, MinLocType &lminloc)
        {
            // Minimise node velocity squared
            double w1 = -NPP_MAXABS_64F;
            for (int icn = 0; icn < NCORN; icn++) {
                double w2 = cnu(iel, icn) * cnu(iel, icn) +
                            cnv(iel, icn) * cnv(iel, icn);
                w1 = BL_MAX(w1, w2);
            }

            w1 = ellength(iel) / BL_MAX(w1, zerocut);
            lminloc.loc = w1 < lminloc.val ? iel : lminloc.loc;
            lminloc.val = w1 < lminloc.val ? w1 : lminloc.val;
        }, MinLoc::reducer(minloc));

    } else {
        // XXX Missing code that can't (or hasn't) been merged
    }

    cudaSync();

    rdt = ale_sf*sqrt(minloc.val);
    idt = minloc.loc;
    sdt = "     ALE";
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
