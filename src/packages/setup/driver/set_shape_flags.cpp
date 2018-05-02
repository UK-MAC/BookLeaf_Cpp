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
#include "packages/setup/driver/set_shape_flags.h"

#include "common/constants.h"
#include "packages/setup/kernel/set_flags.h"
#include "utilities/mix/driver/list.h"
#include "utilities/data/gather.h"
#include "packages/setup/kernel/shapes.h"
#include "packages/setup/types.h"



namespace bookleaf {
namespace setup {
namespace driver {

void
setShapeFlags(
        std::vector<Shape> const &shapes,
        int itarget,
        int iform,
        DataID iflagid,
        Sizes &sizes,
        DataControl &dh,
        Error &err,
        ApplyShapeFunc apply_shape)
{
    int const nforms = shapes.size();
    if (nforms < 1 || (iform < 0 || iform >= nforms)) {
        err.fail("ERROR: failed to find shape");
        return;
    }

    // Gather coordinates
    utils::driver::cornerGather(sizes, DataID::NDX, DataID::SETUP_CNX, dh);
    utils::driver::cornerGather(sizes, DataID::NDY, DataID::SETUP_CNY, dh);

    // Choose function to check whether a point is inside the shape, based on
    // shape type
    InsideFunc inside;
    switch (shapes[iform].type) {
    case Shape::Type::CIRCLE:    inside = kernel::insideCircle;    break;
    case Shape::Type::RECTANGLE: inside = kernel::insideRectangle; break;
    default:
        err.fail("ERROR: Failed to find shape type");
        return;
    }

    apply_shape(itarget, shapes[iform].params, iflagid, sizes, dh, inside);
}



void
applyShapeRegion(
        int itarget,
        double const *param,
        DataID iflagid,
        Sizes &sizes,
        DataControl &data,
        InsideFunc inside)
{
    using constants::NCORN;

    auto flag = data[iflagid].host<int, VarDim>();
    auto cnx  = data[DataID::SETUP_CNX].chost<double, VarDim, NCORN>();
    auto cny  = data[DataID::SETUP_CNY].chost<double, VarDim, NCORN>();

    kernel::setShapeRegion(itarget, param, inside, cnx, cny, flag, sizes.nel);
}



void
applyShapeMaterial(
        int itarget,
        double const *param,
        DataID iflagid,
        Sizes &sizes,
        DataControl &data,
        InsideFunc inside)
{
    using constants::NCORN;

    Error err;

    auto elmat    = data[iflagid].host<int, VarDim>();
    auto cnx      = data[DataID::SETUP_CNX].chost<double, VarDim, NCORN>();
    auto cny      = data[DataID::SETUP_CNY].chost<double, VarDim, NCORN>();

    kernel::setShapeSingleMaterial(itarget, param, inside, cnx, cny, elmat,
            sizes.nel);

    int num_new_mixed_elements, num_new_components;
    kernel::countShapeMixedMaterial(param, inside, cnx, cny, elmat, sizes.nel,
            sizes.nmat, num_new_mixed_elements, num_new_components);

    if (num_new_mixed_elements + num_new_components > 0) {

        // Make space for new mixed material data
        mix::driver::resizeMx(sizes, data, sizes.nmx + num_new_mixed_elements, err);
        mix::driver::resizeCp(sizes, data, sizes.ncp + num_new_components, err);

        auto mxel     = data[DataID::IMXEL].host<int, VarDim>();
        auto mxncp    = data[DataID::IMXNCP].host<int, VarDim>();
        auto mxfcp    = data[DataID::IMXFCP].host<int, VarDim>();
        auto cpmat    = data[DataID::ICPMAT].host<int, VarDim>();
        auto cpprev   = data[DataID::ICPPREV].host<int, VarDim>();
        auto cpnext   = data[DataID::ICPNEXT].host<int, VarDim>();
        auto frvolume = data[DataID::FRVOLUME].host<double, VarDim>();

        kernel::setShapeMixedMaterial(itarget, param, inside, cnx, cny, elmat,
                mxel, mxncp, mxfcp, cpmat, cpprev, cpnext, frvolume, sizes.nel,
                sizes.nmat, sizes.nmx, sizes.ncp);

        sizes.nmx += num_new_mixed_elements;
        sizes.ncp += num_new_components;

        mix::driver::flatten(sizes, data);
    }
}

} // namespace driver
} // namespace setup
} // namespace bookleaf
