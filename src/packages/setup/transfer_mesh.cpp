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
#include "packages/setup/transfer_mesh.h"

#include <cassert>

#include "common/constants.h"
#include "common/error.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "packages/setup/config.h"



namespace bookleaf {
namespace setup {

#define IXm(i, j) (index2D((i), (j), no_l))
#define IXme(i, j) (index2D((i), (j), no_l-1))

void
transferMesh(
        int nel,
        setup::Config &setup_config,
        DataControl &data,
        TimerControl &timers,
        Error &err)
{
    using constants::NCORN;

    ScopedTimer st(timers, TimerID::MESHGEN);

    auto elmat  = data[DataID::IELMAT].host<int, VarDim>();
    auto elreg  = data[DataID::IELREG].host<int, VarDim>();
    auto ndtype = data[DataID::INDTYPE].host<int, VarDim>();
    auto elnd   = data[DataID::IELND].host<int, VarDim, NCORN>();
    auto ndx    = data[DataID::NDX].host<double, VarDim>();
    auto ndy    = data[DataID::NDY].host<double, VarDim>();

    // Transfer information from the now generated mesh region object to the
    // mesh variables in the data controller.
    auto const &mdesc = *setup_config.mesh_descriptor;
    auto const &mdata = *setup_config.mesh_data;

    int const no_l = mdata.dims[0] + 1;
    int const no_k = mdata.dims[1] + 1;

    // Iterate through nodes
    for (int ik = 0; ik < no_k; ik++) {
        for (int il = 0; il < no_l; il++) {

            // Node ordering calculated in mesh renumbering step
            int const ind = mdata.no[IXm(il, ik)];

            // Set node positions
            ndx(ind) = mdata.ss[IXm(il, ik)];
            ndy(ind) = mdata.rr[IXm(il, ik)];

            // Set ndtype (either boundary condition or region value)
            bool const is_bc =
                (il == 0) || (il == no_l-1) ||
                (ik == 0) || (ik == no_k-1);

            if (is_bc) {
                int const bc = mdata.bc[IXm(il, ik)];
                if (bc > 0) {
                    ndtype(ind) = -bc;

                } else {
                    FAIL_WITH_LINE(err, "ERROR: undefined BC at mesh edge");
                    return;
                }

            } else {
                ndtype(ind) = 1; // first (only) mesh region
            }
        }
    }

    // Set element connectivity
    int const i1 = mdata.getNodeOrdering() == 1 ? 3 : 1;
    int const i2 = i1 == 3                      ? 1 : 3;

    for (int ik = 0; ik < no_k - 1; ik++) {
        for (int il = 0; il < no_l - 1; il++) {

            // Element ordering calculated in mesh renumbering step
            int const iel = mdata.eo[IXme(il, ik)];

            // Set element connectivity---use ordered node indices
            elnd(iel, 0)  = mdata.no[IXm(il  , ik  )];
            elnd(iel, 2)  = mdata.no[IXm(il+1, ik+1)];
            elnd(iel, i1) = mdata.no[IXm(il+1, ik  )];
            elnd(iel, i2) = mdata.no[IXm(il  , ik+1)];

            // Set mesh region flags, these are overwritten later unless we are
            // specifying regions and materials by mesh
            elreg(iel) = 0; // first (only) mesh region
        }
    }

    // Set mesh material flags, these are overwritten later unless we are
    // specifying regions and materials by mesh
    for (int iel = 0; iel < nel; iel++) {
        elmat(iel) = mdesc.material;
    }

    // Clean up allocated memory from mesh generation
    setup_config.mesh_data->deallocate();
}

#undef IXm
#undef IXme

} // namespace setup
} // namespace bookleaf
