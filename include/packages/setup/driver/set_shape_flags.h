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
#ifndef BOOKLEAF_PACKAGES_SETUP_DRIVER_SET_FLAGS_H
#define BOOKLEAF_PACKAGES_SETUP_DRIVER_SET_FLAGS_H

#include "common/sizes.h"
#include "common/data_control.h"

#include "packages/setup/types.h"



namespace bookleaf {

struct Sizes;

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
        ApplyShapeFunc apply_shape);

void
applyShapeRegion(
        int itarget,
        double const *param,
        DataID iflagid,
        Sizes &sizes,
        DataControl &data,
        InsideFunc inside);

void
applyShapeMaterial(
        int itarget,
        double const *param,
        DataID iflagid,
        Sizes &sizes,
        DataControl &data,
        InsideFunc inside);

} // namespace driver
} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_DRIVER_SET_FLAGS_H
