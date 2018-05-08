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
#include "packages/setup/kernel/set_flags.h"

#include "common/sizes.h"
#include "common/error.h"
#include "common/data_id.h"
#include "common/data_control.h"

#include "utilities/geometry/geometry.h"
#include "utilities/mix/kernel/list.h"

#include "packages/setup/types.h"
#include "packages/setup/kernel/shapes.h"



namespace bookleaf {
namespace setup {
namespace kernel {
namespace flag {

void
setCellFlags(
        int nel,
        int iflag,
        int itest,
        int nproc,
        ConstView<int, VarDim> test,
        View<int, VarDim>      flag,
        Error &err)
{
    // Set flag for cell index
    if (nproc > 1) {
        setFlagIf(nel, iflag, itest, test, flag);

    } else {
        if (itest < 0 || itest >= (int) flag.size()) {
            err.fail("ERROR: incorrect cell index in flag::cell");
            return;
        }

        flag(itest) = iflag;
    }
}

} // namespace flag

void
setFlag(
        int nel,
        int iflag,
        View<int, VarDim> flag)
{
    for (int iel = 0; iel < nel; iel++) {
        flag(iel) = iflag;
    }
}



void
setFlagIf(
        int nel,
        int iflag,
        int itest,
        ConstView<int, VarDim> test,
        View<int, VarDim>      flag)
{
    for (int iel = 0; iel < nel; iel++) {
        if (test(iel) == itest) {
            flag(iel) = iflag;
        }
    }
}



void
setShapeRegion(
        int ireg,
        double const *shape_param,
        InsideFunc inside_shape,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                elreg,
        int nel)
{
    using constants::NDIM;

    for (int iel = 0; iel < nel; iel++) {
        double _point[NDIM];
        View<double, NDIM> point(_point);

        geometry::kernel::getCentroid(iel, cnx, cny, point);

        // If the centroid of the element is within the shape then set the
        // region.
        if (inside_shape(shape_param, _point)) {
            elreg(iel) = ireg;
        }
    }
}



void
setShapeSingleMaterial(
        int itarget,
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                flag,
        int nel)
{
    for (int iel = 0; iel < nel; iel++) {

        // If the element is entirely within the shape, assign a single material
        int const count = intersect(param, iel, cnx, cny, inside);

        if (count == NCORN) {
            flag(iel) = itarget;
        }
    }
}



void
countShapeMixedMaterial(
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<int, VarDim>           flag,
        int nel,
        int nmat,
        int &num_new_mixed_elements,
        int &num_new_components)
{
    num_new_mixed_elements = 0;
    num_new_components     = 0;

    for (int iel = 0; iel < nel; iel++) {

        // If the element is only partially within the shape, count how many new
        // mixed elements and components we will need to add.
        int const count = intersect(param, iel, cnx, cny, inside);

        if (count > 0 && count < NCORN) {

            // XXX This logic assumes that a background material has already
            //     been set.

            int j = flag(iel);
            if (j >= 0 && j < nmat) {
                num_new_mixed_elements++;
                num_new_components++;
            }

            num_new_components++;
        }
    }
}



void
setShapeMixedMaterial(
        int itarget,
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                elmat,
        View<int, VarDim>                mxel,
        View<int, VarDim>                mxncp,
        View<int, VarDim>                mxfcp,
        View<int, VarDim>                cpmat,
        View<int, VarDim>                cpprev,
        View<int, VarDim>                cpnext,
        View<double, VarDim>             frvolume,
        int nel,
        int nmat,
        int nmx,
        int ncp)
{
    Error err;

    int imx = nmx;
    int icp = ncp;

    for (int iel = 0; iel < nel; iel++) {

        // If the element is only partially within the shape, update the mixed
        // material data structures.
        int const count = intersect(param, iel, cnx, cny, inside);

        if (count > 0 && count < NCORN) {

            // XXX This logic assumes that a background material has already
            //     been set.

            // Calculate volume fraction
            double const vf = subdivide(0, param, iel, cnx, cny, inside);

            // Get the current material index for this element
            int j = elmat(iel);
            assert(j <= nmat);

            // If the element is currently single-material, we will add a new
            // mixed-material element in the next branch, otherwise we will use
            // the existing mixed-material element.
            int const cur_imx = j >= 0 ? imx : -(j+1);

            // If the element is currently single-material, upgrade it to
            // mixed-material.
            if (j >= 0 && j < nmat) {
                int const imat = j;
                mix::kernel::addEl(iel, cur_imx, elmat, mxel, mxncp, mxfcp);
                imx++; // Added a new mixed-material element

                // Add component corresponding to the original single-material.
                mix::kernel::addCp(cur_imx, icp, mxfcp, mxncp, cpprev, cpnext);
                cpmat(icp) = imat;
                frvolume(icp) = 1.0 - vf;
                icp++;
            }

            // Shouldn't be able to get into this branch if the background
            // material has been set.
            if (j == nmat) {
                FAIL_WITH_LINE(err, "ERROR: logic error");
                return;
            }

            // Add component corresponding to the new shape material.
            mix::kernel::addCp(cur_imx, icp, mxfcp, mxncp, cpprev, cpnext);
            cpmat(icp) = itarget;
            frvolume(icp) = vf;
            icp++;
        }
    }
}

} // namespace kernel
} // namespace setup
} // namespace bookleaf
