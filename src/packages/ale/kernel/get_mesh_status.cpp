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
#include "packages/ale/kernel/get_mesh_status.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
getMeshStatus(
        int nnd,
        bool zeul,
        View<int, VarDim> ndstatus)
{
    if (zeul) {
        for (int ind = 0; ind < nnd; ind++) {
            ndstatus(ind) = 2;
        }

    } else {
        // XXX Missing code that can't be (or hasn't been) merged.
    }
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
