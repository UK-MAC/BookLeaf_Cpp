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

#define IXm(i, j) (index2D((i-1), (j-1), no_l))

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

    //int *istore = new int[nnd];
    //std::fill(istore, istore + nnd, 0);

    double r1, r2, r3, s1, s2, s3;

    // Initialise
    int nod_count = -1;

    int const nreg = setup_config.mesh_regions.size();
    for (int ireg = 0; ireg < nreg; ireg++) {
        MeshRegion const &mr = setup_config.mesh_regions[ireg];

        int const no_l = mr.dims[0] + 1;
        int const no_k = mr.dims[1] + 1;

        int l1 = 1;
        int l2 = no_l;
        int k1 = 1;
        int k2 = no_k;

        // Coords + node type
        for (int kk = k1; kk <= k2; kk++) {
            for (int ll = l1; ll <= l2; ll++) {
                if (ll == l1 || ll == l2 || kk == k1 || kk == k2) {
                    if (!mr.merged[IXm(ll, kk)]) {
                        nod_count++;
                        ndx(nod_count) = mr.ss[IXm(ll, kk)];
                        ndy(nod_count) = mr.rr[IXm(ll, kk)];
                        int i1 = mr.bc[IXm(ll, kk)];
                        if (i1 > 0) {
                            ndtype(nod_count) = -i1;
                            //istore[nod_count] = ireg;
                        } else {
                            err.fail("ERROR: undefined BC at region edge");
                            return;
                        }
                    }

                } else {
                    nod_count++;
                    ndx(nod_count) = mr.ss[IXm(ll, kk)];
                    ndy(nod_count) = mr.rr[IXm(ll, kk)];
                    ndtype(nod_count) = ireg+1;
                }
            }
        }

        // Connectivity
        r1 = mr.rr[IXm(2, 1)] - mr.rr[IXm(1, 1)];
        r2 = mr.rr[IXm(1, 2)] - mr.rr[IXm(1, 1)];
        r3 = mr.rr[IXm(2, 2)] - mr.rr[IXm(1, 1)];
        s1 = mr.ss[IXm(2, 1)] - mr.ss[IXm(1, 1)];
        s2 = mr.ss[IXm(1, 2)] - mr.ss[IXm(1, 1)];
        s3 = mr.ss[IXm(2, 2)] - mr.ss[IXm(1, 1)];

        int i1, i2;
        if (((r1*s2-r2*s1) > 0.) || ((r1*s3-r3*s1) > 0.) ||
                ((r3*s2-r2*s3) > 0.)) {
            i1 = 3;
            i2 = 1;
        } else {
            i1 = 1;
            i2 = 3;
        }

        int ele_count = -1;
        for (int kk = k1; kk <= k2 - 1; kk++) {
            for (int ll = l1; ll <= l2 - 1; ll++) {
                ele_count++;
                elnd(ele_count, 0)  = (ll-1) + (kk-1)*l2;
                elnd(ele_count, 2)  = (ll)   + (kk)  *l2;
                elnd(ele_count, i1) = elnd(ele_count, 0)+1;
                elnd(ele_count, i2) = elnd(ele_count, 2)-1;
                elreg(ele_count) = ireg;
            }
        }

        // Clean up allocated memory from mesh region (allocated by
        // setup::generateMesh())
        delete[] mr.rr;
        delete[] mr.ss;
        delete[] mr.merged;
        delete[] mr.bc;

    } // for ireg < nreg

    // Material no.
    for (int iele = 0; iele < nel; iele++) {
        int ireg = elreg(iele);
        elmat(iele) = setup_config.mesh_regions[ireg].material;
    }

    // No longer need mesh region data
    setup_config.mesh_regions.clear();

    //delete[] istore;
}

#undef IXm

} // namespace setup
} // namespace bookleaf
