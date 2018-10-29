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
#include "packages/setup/renumber_mesh.h"

#include <cassert>
#include <iomanip>
#include <numeric>
#include <algorithm>
#include <cmath>
//#include <random>

#include "packages/setup/config.h"
#include "common/data_control.h"
#include "common/timer_control.h"
#include "common/sizes.h"
#include "common/constants.h"



#define IX(i, j) (index2D((i), (j), (dims[0]+1)))
#define IXe(i, j) (index2D((i), (j), (dims[0])))

namespace bookleaf {
namespace setup {
namespace {

void
remapCellIndicators(
        setup::Config &setup_config)
{
    auto const &mdata = *setup_config.mesh_data;
    int const dims[2] = { mdata.dims[0], mdata.dims[1] };

    // Remap region indicators
    for (auto &region : setup_config.regions) {
        if (region.type == Indicator::Type::CELL) {

            // This is the value set by the user in the input deck
            int const value = region.value;

            // Calculate the logical mesh indices for this "canonical" cell
            // index.
            int const ik = value / dims[0];
            int const il = value % dims[0];

            // Get the new global number and update the region
            int const ielg = mdata.eo[IXe(il, ik)];
            region.value = ielg;
        }
    }

    // Remap material indicators
    for (auto &material : setup_config.materials) {
        if (material.type == Indicator::Type::CELL) {

            // This is the value set by the user in the input deck
            int const value = material.value;

            // Calculate the logical mesh indices for this "canonical" cell
            // index.
            int const ik = value / dims[0];
            int const il = value % dims[0];

            // Get the new global number and update the region
            int const ielg = mdata.eo[IXe(il, ik)];
            material.value = ielg;
        }
    }
}

} // namespace

void
renumberMesh(
        bookleaf::Config const &config __attribute__((unused)),
        setup::Config &setup_config,
        TimerControl &timers,
        Error &err __attribute__((unused)))
{
    ScopedTimer st(timers, TimerID::MESHRENUM);

    auto &mdata = *setup_config.mesh_data;

    // Mesh dimensions
    int const dims[2] = {
        mdata.dims[0],
        mdata.dims[1]
    };

    int const side __attribute__((unused)) = dims[1] >= dims[0] ? 1 : 0;
    int const nelg = dims[0] * dims[1];
    int const nndg = (dims[0]+1) * (dims[1]+1);

    int *nn = mdata.nn;
    int *en = mdata.en;
    int *no = mdata.no;
    int *eo = mdata.eo;

    // Initialise node and element numberings to default
    int iel = 0;
    for (int kk = 0; kk < dims[1]+1; kk++) {
        for (int ll = 0; ll < dims[0]+1; ll++) {
            int const ind = ll + kk * (dims[0]+1);
            assert(ind < nndg);

            nn[IX(ll, kk)] = ind;
            if (kk < dims[1] && ll < dims[0]) en[IXe(ll, kk)] = iel++;
        }
    }

    // XXX can insert code to alter numberings here

    // Initialise node and element orderings
    std::iota(no, no+nndg, 0);
    std::iota(eo, eo+nelg, 0);
    std::sort(no, no+nndg, [&](int i, int j) { return nn[i] < nn[j]; });
    std::sort(eo, eo+nelg, [&](int i, int j) { return en[i] < en[j]; });

    // After renumbering the mesh, we may need to update any region/material
    // indicators targeting specific cells (as their indices may have changed)
    remapCellIndicators(setup_config);
}

} // namespace setup
} // namespace bookleaf

#undef IX
#undef IXe
