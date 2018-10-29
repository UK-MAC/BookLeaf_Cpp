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
#include "packages/ale/kernel/get_mesh_velocity.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif



namespace bookleaf {
namespace ale {
namespace kernel {

void
getMeshVelocity(
        int nnd,
        bool zeul,
        View<double, VarDim> ndu,
        View<double, VarDim> ndv)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    if (zeul) {
        RAJA::forall<RAJA_POLICY>(
                RAJA::RangeSegment(0, nnd),
                BOOKLEAF_DEVICE_LAMBDA (int const ind)
        {
            ndu(ind) = -ndu(ind);
            ndv(ind) = -ndv(ind);
        });

    } else {
        // XXX Missing code that can't (or hasn't) been merged.
    }
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
