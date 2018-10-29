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
#include "packages/init/kernel.h"

#include <cassert>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>

#include "common/constants.h"
#include "common/sizes.h"
#include "common/data_control.h"
#include "utilities/data/sort.h"



// XXX(timrlaw): these connectivity routines are highly confusing, but they
//               seem to validate vs. fortran now
namespace bookleaf {
namespace init {
namespace kernel {

void
getElementConnectivity(
        ConstView<int, VarDim, NCORN> elnd,
        View<int, VarDim, NFACE>      elel,
        int nel)
{
    // XXX(timrlaw): Need to be careful here---had issues with face IDs
    //               overflowing 4-byte integers. SizeType should be 8-byte
    static_assert(sizeof(SizeType) >= 8, "SizeType too small for connectivity");

    // Initialise
    int const nsz = NFACE * nel;
    std::unique_ptr<SizeType[]> iConn(new SizeType[nsz]);
    std::unique_ptr<SizeType[]> iUind(new SizeType[nsz]);
    std::unique_ptr<SizeType[]> iWork1(new SizeType[nsz]);
    std::unique_ptr<SizeType[]> iWork2(new SizeType[nsz]);

    View<SizeType, VarDim> vconn(iConn.get(), nsz);
    View<SizeType, VarDim> vuind(iUind.get(), nsz);
    View<SizeType, VarDim> vwork1(iWork1.get(), nsz);
    View<SizeType, VarDim> vwork2(iWork2.get(), nsz);

    // Transpose elnd into work1 and work2, so that we have two lists of
    // element-node connectivity, offset by nel.
    for (int iel = 0; iel < nel; iel++) {
        int const i2 = iel + nel*1;
        int const i3 = iel + nel*2;
        int const i4 = iel + nel*3;

        vwork1(iel) = (SizeType) elnd(iel, 0);
        vwork2(iel) = (SizeType) elnd(iel, 1);
        vwork1(i2) = (SizeType) elnd(iel, 1);
        vwork2(i2) = (SizeType) elnd(iel, 2);
        vwork1(i3) = (SizeType) elnd(iel, 2);
        vwork2(i3) = (SizeType) elnd(iel, 3);
        vwork1(i4) = (SizeType) elnd(iel, 3);
        vwork2(i4) = (SizeType) elnd(iel, 0);

        vconn(iel) = vconn(i2) = vconn(i3) = vconn(i4) = (SizeType) iel;
    }

    // Compute unique IDs for each adjacent node pair (equivalent to each face).
    // The ID will be the same regardless of the order of nodes, which allows us
    // to find matching faces by finding matching IDs.
    //
    //   ID = lesser_node_num * max_local_node_num + greater_node_num
    //
    SizeType elnd_max = 0;
    for (int iel = 0; iel < nel; iel++) {
        for (int icn = 0; icn < NCORN; icn++) {
            elnd_max = (SizeType) std::max((SizeType) elnd(iel, icn), elnd_max);
        }
    }

    for (int i = 0; i < nsz; i++) {
        SizeType const work1 = vwork1(i);
        SizeType const work2 = vwork2(i);

        SizeType const work_max = std::max(work1, work2);
        SizeType const work_min = std::min(work1, work2);

        vuind(i) = (work_min * elnd_max) + work_max;

        // Check for overflow
        if (vuind(i) > (((elnd_max-1) * elnd_max) + elnd_max)) {
            assert(false && "face ID overflow");
        }
    }

    // Determine sorted order of face IDs (in work1)
    utils::kernel::sortIndices<SizeType, SizeType>(vuind, vwork1, nsz);

    // Find matching faces, store match offsets in vwork2
    int num_matches = 0;
    for (int i = 0; i < nsz - 1; i++) {
        SizeType const cur  = vwork1(i);
        SizeType const next = vwork1(i + 1);
        if (vuind(cur) == vuind(next)) vwork2(num_matches++) = i;
    }

    // Reorder elements numbers in line with faces, so we can figure out which
    // elements share a face
    for (int i = 0; i < nsz; i++) {
        vuind(i) = vwork1(i);
    }

    utils::kernel::reorder<SizeType, SizeType>(vwork1, vconn, nsz);

    // Insert match element numbers into connectivity table (work1)
    for (int i = 0; i < nsz; i++) {
        vwork1(i) = (SizeType) -1;
    }

    for (int i = 0; i < num_matches; i++) {
        SizeType const match_offset = vwork2(i);
        vwork1(vuind(match_offset))   = vconn(match_offset+1);
        vwork1(vuind(match_offset+1)) = vconn(match_offset);
    }

    // Copy to elel (element-element connectivity). Note interchanged loop order
    // to untranspose data
    int work_index = 0;
    for (int j = 0; j < NFACE; j++) {
        for (int iel = 0; iel < nel; iel++) {
            elel(iel, j) = vwork1(work_index++);
        }
    }
}



void
getFaceConnectivity(
        ConstView<int, VarDim, NFACE> elel,
        View<int, VarDim, NFACE>      elfc,
        int nel)
{
    for (int iel = 0; iel < nel; iel++) {
        elfc(iel, 0) = -1;
        elfc(iel, 1) = -1;
        elfc(iel, 2) = -1;
        elfc(iel, 3) = -1;
    }

    // For each element
    for (int iel = 0; iel < nel; iel++) {

        // For each adjacent element
        for (int j = 0; j < NFACE; j++) {
            int const ineigh = elel(iel, j);
            if (ineigh > -1) {

                // Store face index that connects to current element
                for (int k = 0; k < NFACE; k++) {
                    if (elel(ineigh, k) == iel) {
                        elfc(iel, j) = k;
                        break;
                    }
                }
            }
        }
    }
}



void
correctConnectivity(
        int ncell,
        View<int, VarDim, NFACE> elel,
        View<int, VarDim, NFACE> elfc)
{
    for (int iel = 0; iel < ncell; iel++) {
        for (int i1 = 0; i1 < NFACE/2; i1++) {
            int i2 = i1 + 2;
            if (elel(iel, i1) == -1) {
                elel(iel, i1) = iel;
                elfc(iel, i1) = i2;
            }

            if (elel(iel, i2) == -1) {
                elel(iel, i2) = iel;
                elfc(iel, i2) = i1;
            }
        }
    }
}



void
getNodeElementMappingSizes(
        ConstView<int, VarDim, NCORN> elnd,
        View<int, VarDim>             ndeln,
        View<int, VarDim>             ndelf,
        int nnd,
        int nel)
{
    for (int ind = 0; ind < nnd; ind++) {
        ndeln(ind) = 0;
    }

    // Get number of elements per node (XXX not thread-safe)
    for (int iel = 0; iel < nel; iel++) {
        for (int icn = 0; icn < NCORN; icn++) {
            int const ind = elnd(iel, icn);
            ndeln(ind)++;
        }
    }

    // Get mapping offsets
    ndelf(0) = 0;
    for (int ind = 1; ind < nnd; ind++) {
        ndelf(ind) = ndelf(ind-1) + ndeln(ind-1);
    }
}



void
getNodeElementMapping(
        ConstView<int, VarDim, NCORN> elnd,
#ifdef BOOKLEAF_DEBUG
        ConstView<int, VarDim>        ndeln,
#else
        ConstView<int, VarDim>        ndeln __attribute__((unused)),
#endif
        ConstView<int, VarDim>        ndelf,
        View<int, VarDim>             ndel,
        int nnd,
        int nel)
{
    std::unique_ptr<int[]> ndoffsets(new int[nnd]);
    std::fill(&ndoffsets[0], &ndoffsets[0] + nnd, 0);

    for (int iel = 0; iel < nel; iel++) {
        for (int icn = 0; icn < NCORN; icn++) {
            int const ind = elnd(iel, icn);

            int const start  = ndelf(ind);
            int const offset = ndoffsets[ind];

            ndel(start+offset) = iel;

            ndoffsets[ind]++;
        }
    }

#ifdef BOOKLEAF_DEBUG
    // Check the lengths match up
    for (int ind = 0; ind < nnd; ind++) {
        if (ndoffsets[ind] != ndeln(ind)) {
            assert(false && "unhandled error");
        }
    }
#endif
}



void
elMass(
        int nel,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elvolume,
        ConstView<double, VarDim, NCORN> cnwt,
        View<double, VarDim>             elmass,
        View<double, VarDim, NCORN>      cnmass)
{
    for (int i = 0; i < nel; i++) {
        elmass(i) = eldensity(i) * elvolume(i);

        for (int j = 0; j < NCORN; j++) {
            cnmass(i, j) = eldensity(i) * cnwt(i, j);
        }
    }
}



void
mxMass(
        int ncp,
        ConstView<double, VarDim> mxdensity,
        ConstView<double, VarDim> mxvolume,
        View<double, VarDim>      mxmass)
{
    for (int i = 0; i < ncp; i++) {
        mxmass(i) = mxdensity(i) * mxvolume(i);
    }
}



void
nodeType(
        ConstView<int, VarDim, NCORN> elnd,
        View<int, VarDim>             ndtype,
        int nel,
        int nnd)
{
    int _nodes[NCORN] = {0};
    View<int, NCORN> nodes(_nodes, NCORN);

    for (int iel = 0; iel < nel; iel++) {
        for (int j = 0; j < NCORN; j++) {
            nodes(j) = elnd(iel, j);
        }

        int count = std::count_if(
                &nodes(0), &nodes(0) + NCORN,
                [](int node) { return node < 0; });

        if (count == 3) {
            int ii = 0;
            for ( ; ii < NCORN; ii++) {
                if (ndtype(nodes(ii)) > 0) break;
            }
            ii = (ii + 2) % NCORN;

            int jj = nodes(ii);
            if (jj <= nnd) {
                int j1 = nodes((ii + 1) % NCORN);
                int j2 = nodes((ii + 3) % NCORN);
                if (((ndtype(j1) == -2) && (ndtype(j2) == -1)) ||
                    ((ndtype(j2) == -2) && (ndtype(j1) == -1))) {
                    ndtype(jj) = -3;
                }
            }
        }
    }
}

} // namespace kernel
} // namespace init
} // namespace bookleaf
