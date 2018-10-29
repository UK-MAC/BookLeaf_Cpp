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
#include "packages/setup/partition_mesh.h"

#include <cassert>
#include <algorithm>
#include <fstream>

#include <typhon.h>

#ifdef BOOKLEAF_PARMETIS_SUPPORT
#include <parmetis.h>
#endif

#include "common/error.h"
#include "common/sizes.h"
#include "common/constants.h"
#include "common/cmd_args.h"
#include "packages/setup/distribute_mesh.h"
#include "common/timer_control.h"
#include "common/config.h"
#include "packages/setup/config.h"
#include "common/data_control.h"

#include "utilities/comms/config.h"
#include "utilities/comms/partition.h"



namespace bookleaf {
namespace setup {
namespace {

void
computeConnectivity(
        setup::Config const &setup_config,
        int const dims[2],
        int const side,
        int const slice[2],
        int *_connectivity,
        Error &err)
{
    using constants::NDAT;

    if (_connectivity == nullptr) {
        FAIL_WITH_LINE(err, "ERROR: connectivity should be preallocated");
        return;
    }

    auto const &mdesc = *setup_config.mesh_descriptor;
    auto const &mdata = *setup_config.mesh_data;

    int const no_l = dims[0] + 1;
    #define IXmn(i, j) (index2D((i), (j), no_l))
    #define IXme(i, j) (index2D((i), (j), no_l-1))

    // Set conndata
    int *conn_data = _connectivity;
    #define IXconn(i, j) (index2D((j), (i), NDAT))

    int const in1 = mdata.getNodeOrdering() == 1 ? 6 : 4;
    int const in2 = in1 == 6                     ? 4 : 6;

    int const k_lo = side == 1 ? slice[0] : 0;
    int const k_hi = side == 1 ? slice[1] : dims[1];
    int const l_lo = side == 0 ? slice[0] : 0;
    int const l_hi = side == 0 ? slice[1] : dims[0];

    //
    // Connectivity data structure:
    //
    //  [0]: global element index
    //  [1]: mesh region
    //  [2]: mesh material
    //  [3]: global node index 0
    //  [4]: global node index 1
    //  [5]: global node index 2
    //  [6]: global node index 3
    //
    int iel = 0;
    for (int ik = k_lo; ik < k_hi; ik++) {
        for (int il = l_lo; il < l_hi; il++) {

            // Global element index
            conn_data[IXconn(iel, 0)] = mdata.eo[IXme(il, ik)];

            // Mesh region/material
            conn_data[IXconn(iel, 1)] = 0;
            conn_data[IXconn(iel, 2)] = mdesc.material;

            // Global node indices
            conn_data[IXconn(iel, 3)]   = mdata.no[IXmn(il  , ik  )];
            conn_data[IXconn(iel, 5)]   = mdata.no[IXmn(il+1, ik+1)];
            conn_data[IXconn(iel, in1)] = mdata.no[IXmn(il+1, ik  )];
            conn_data[IXconn(iel, in2)] = mdata.no[IXmn(il  , ik+1)];

            iel++;
        }
    }

    #undef IXmn
    #undef IXme
    #undef IXconn
}



void
meshNodalData(
        setup::Config &setup_config,
        int nnd,
        ConstView<int, VarDim> ndlocglob,
        View<int, VarDim>      ndtype,
        View<double, VarDim>   ndx,
        View<double, VarDim>   ndy,
        Error &err)
{
    auto const &mdata = *setup_config.mesh_data;

    // Extents
    int const no_l = mdata.dims[0] + 1;
    int const no_k = mdata.dims[1] + 1;

    #define IXmn(i, j) (index2D(i, j, no_l))

    // Check that ndlocglob is sorted
    bool const sorted = std::is_sorted(&ndlocglob(0), &ndlocglob(nnd));
    if (!sorted) {
        FAIL_WITH_LINE(err, "ERROR: node number array unsorted");
        return;
    }

    // Check that ndlocglob doesn't contain duplicates
    int prev = ndlocglob(0);
    for (int ind = 1; ind < nnd; ind++) {
        if (ndlocglob(ind) == prev) {
            FAIL_WITH_LINE(err, "ERROR: duplicates in node number array");
            return;
        }

        prev = ndlocglob(ind);
    }

    // Co-ordinates and BCs
    int num_found = 0;
    for (int ik = 0; ik < no_k; ik++) {
        for (int il = 0; il < no_l; il++) {

            // Get the global node number
            int const indg = mdata.no[IXmn(il, ik)];

            // See if this node ended up in this rank's partition, ignore if not
            auto it = std::lower_bound(&ndlocglob(0), &ndlocglob(nnd), indg);
            if (it == &ndlocglob(nnd) || *it != indg) continue;

            int const ind = it - &ndlocglob(0);
            num_found++;

            // Set node positions
            ndx(ind) = mdata.ss[IXmn(il, ik)];
            ndy(ind) = mdata.rr[IXmn(il, ik)];

            // Set ndtype (either boundary condition or region value)
            bool const is_bc =
                (il == 0) || (il == no_l-1) ||
                (ik == 0) || (ik == no_k-1);

            if (is_bc) {
                int const bc = mdata.bc[IXmn(il, ik)];
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

    // Check all mesh accounted for
    if (num_found != nnd) {
        std::cerr << num_found << "/" << nnd << "\n";
        FAIL_WITH_LINE(err, "ERROR: Missing nodes on process");
        return;
    }

    #undef IXmn

    // Deallocate mesh data
    setup_config.mesh_data->deallocate();
}

} // namespace

void
partitionMesh(
        bookleaf::Config const &config,
        setup::Config &setup_config,
        Sizes &sizes,
        TimerControl &timers,
        TimerID timer_id,
        DataControl &data,
        Error &err)
{
    using constants::NDAT;

    ScopedTimer st(timers, timer_id);

    comms::Comm &comm = *config.comms->spatial;

    auto const &mdata = *setup_config.mesh_data;

    // Mesh dimensions
    int const dims[2] = { mdata.dims[0], mdata.dims[1] };

    // Initial partitioning
    int side;
    int slice[2];
    int nel;
    comms::initialPartition(dims, comm, side, slice, nel);

    // Compute connectivity
    int *connectivity = new int[NDAT * nel];
    int conn_dims[2] = { NDAT, nel };
    computeConnectivity(setup_config, dims, side, slice, connectivity, err);
    if (err.failed()) return;

    // Improve partitioning
    int *_coldata = new int[nel];
    View<int, VarDim> coldata(_coldata, nel);
    comms::improvePartition(dims, side, slice, nel, comm, connectivity, coldata,
            err);
    if (err.failed()) return;

    // Distribute mesh data to partition
    distributeMesh(connectivity, conn_dims, coldata.data(), config, sizes, data,
            err);
    if (err.failed()) return;

    delete[] connectivity;
    delete[] _coldata;

    // Set nodal mesh data
    auto ndlocglob = data[DataID::INDLOCGLOB].chost<int, VarDim>();
    auto ndtype    = data[DataID::INDTYPE].host<int, VarDim>();
    auto ndx       = data[DataID::NDX].host<double, VarDim>();
    auto ndy       = data[DataID::NDY].host<double, VarDim>();

    meshNodalData(setup_config, sizes.nnd, ndlocglob, ndtype, ndx, ndy, err);
    if (err.failed()) return;
}

} // namespace setup
} // namespace bookleaf
