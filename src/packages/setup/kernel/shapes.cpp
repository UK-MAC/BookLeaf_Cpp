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
#include "packages/setup/kernel/shapes.h"

#include "common/view.h"

#include "utilities/geometry/geometry.h"
#include "utilities/mix/driver/list.h"



namespace bookleaf {
namespace setup {
namespace kernel {

bool
insideCircle(
        double const *shape_param,
        double const *point)
{
    // shape_param should contain:
    //  [0] = centre x
    //  [1] = centre y
    //  [2] = radius

    double const x = point[0] - shape_param[0];
    double const y = point[1] - shape_param[1];
    double const r = shape_param[2];

    return (x*x + y*y) <= (r*r);
}



bool
insideRectangle(
        double const *shape_param,
        double const *point)
{
    // shape_param should contain:
    //  [0] = x1
    //  [1] = y1
    //  [2] = x2
    //  [2] = y2

    return point[0] > shape_param[0] &&
           point[1] > shape_param[1] &&
           point[0] < shape_param[2] &&
           point[1] < shape_param[3];
}

} // namespace kernel
} // namespace setup
} // namespace bookleaf
