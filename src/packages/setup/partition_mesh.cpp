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
        int const nel,
        int *_connectivity,
        Error &err)
{
    using constants::NDAT;

    if (_connectivity == nullptr) {
        FAIL_WITH_LINE(err, "ERROR: connectivity should be preallocated");
        return;
    }

    std::vector<MeshRegion> const &mesh_regions = setup_config.mesh_regions;

    // Assume mesh has only 1 region
    MeshRegion const &mr = mesh_regions[0];
    int const no_l = dims[0] + 1;
    #define IX(i, j) (index2D((i), (j), no_l))

    // Set conndata
    View<int, VarDim, NDAT> conn_data(_connectivity, nel);

    double r1 = mr.rr[IX(1, 0)] - mr.rr[IX(0, 0)];
    double r2 = mr.rr[IX(0, 1)] - mr.rr[IX(0, 0)];
    double r3 = mr.rr[IX(1, 1)] - mr.rr[IX(0, 0)];
    double s1 = mr.ss[IX(1, 0)] - mr.ss[IX(0, 0)];
    double s2 = mr.ss[IX(0, 1)] - mr.ss[IX(0, 0)];
    double s3 = mr.ss[IX(1, 1)] - mr.ss[IX(0, 0)];

    int in1, in2;
    if (((r1*s2-r2*s1) > 0.) || ((r1*s3-r3*s1) > 0.) || ((r3*s2-r2*s3) > 0.)) {
        in1 = 6;
        in2 = 4;
    } else {
        in1 = 4;
        in2 = 6;
    }

    //
    // conndata structure:
    //
    //  - global element index
    //  - mesh region
    //  - material
    //  - node 0
    //  - node 1
    //  - node 2
    //  - node 3
    //
    int iel = 0;
    if (side == 1) {
        for (int kk = slice[0]; kk < slice[1]; kk++) {
            for (int ll = 0; ll < dims[0]; ll++) {
                assert(iel < nel);

                conn_data(iel, 0) = iel + (dims[0] * slice[0]);
                conn_data(iel, 1) = 0;
                conn_data(iel, 2) = mr.material;

                // Global node indices
                conn_data(iel, 3)   = ll + kk * (dims[0] + 1);
                conn_data(iel, 5)   = (ll+1) + (kk+1) * (dims[0] + 1);
                conn_data(iel, in1) = conn_data(iel, 3) + 1;
                conn_data(iel, in2) = conn_data(iel, 5) - 1;

                iel++;
            }
        }

    } else {
        for (int kk = 0; kk < dims[1]; kk++) {
            for (int ll = slice[0]; ll < slice[1]; ll++) {
                assert(iel < nel);

                conn_data(iel, 0) = iel + kk * dims[0] + slice[0];
                conn_data(iel, 1) = 0;
                conn_data(iel, 2) = mr.material;

                // Global node indices
                conn_data(iel, 3)   = ll + kk * (dims[0] + 1);
                conn_data(iel, 5)   = (ll+1) + (kk+1) * (dims[0] + 1);
                conn_data(iel, in1) = conn_data(iel, 3) + 1;
                conn_data(iel, in2) = conn_data(iel, 5) - 1;

                iel++;
            }
        }
    }
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
    // Local decomposition, assume nreg=1
    assert(setup_config.mesh_regions.size() == 1);
    MeshRegion const &mr = setup_config.mesh_regions[0];

    // Extents
    int const l1 = 0;
    int const l2 = mr.dims[0] + 1;
    int const k1 = 0;
    int const k2 = mr.dims[1] + 1;
    int const ii = std::max(l2, k2);

    #define IXm(i, j) (index2D(i, j, mr.dims[0] + 1))

    // Co-ordinates and BCs
    int icount = 0;
    for (int kk = k1; kk < k2; kk++) {
        for (int ll = l1; ll < l2; ll++) {
            int const indg = ii == k2 ? ll + kk * l2 : kk + ll * k2;
            assert(indg >= 0);

            bool const found = std::binary_search(&ndlocglob(0), &ndlocglob(nnd), indg);
            if (found) {
                bool const edge = (ll == l1) || (ll == l2-1) ||
                                  (kk == k1) || (kk == k2-1);
                if (edge) {
                    if (!mr.merged[IXm(ll, kk)]) {
                        ndx(icount) = mr.ss[IXm(ll, kk)];
                        ndy(icount) = mr.rr[IXm(ll, kk)];
                        if (mr.bc[IXm(ll, kk)] > 0) {
                            ndtype(icount) = -mr.bc[IXm(ll, kk)];
                        } else {
                            FAIL_WITH_LINE(err,
                                    "ERROR: undefined BC at mesh edge");
                            return;
                        }

                        icount++;
                    }

                } else {
                    ndx(icount) = mr.ss[IXm(ll, kk)];
                    ndy(icount) = mr.rr[IXm(ll, kk)];
                    ndtype(icount) = 1;
                    icount++;
                }
            } // if (*it == indg)
        } // for (int ll = l1; ll < l2; ll++)
    } // for (int kk = k1; kk < k2; kk++)

    // Check all mesh accounted for
    if (icount != nnd) {
        FAIL_WITH_LINE(err, "ERROR: Missing nodes on process");
        return;
    }

    #undef IXm

    // Destroy reg
    delete[] mr.rr;
    delete[] mr.ss;
    delete[] mr.merged;
    delete[] mr.bc;
    setup_config.mesh_regions.clear();
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

    // Mesh dimensions
    int const dims[2] = {
        setup_config.mesh_regions[0].dims[0],
        setup_config.mesh_regions[0].dims[1]
    };

    // Initial partitioning
    int side;
    int slice[2];
    int nel;
    comms::initialPartition(dims, comm, side, slice, nel);

    // Compute connectivity
    int *connectivity = new int[NDAT * nel];
    int conn_dims[2] = { NDAT, nel };
    computeConnectivity(setup_config, dims, side, slice, nel, connectivity, err);
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
