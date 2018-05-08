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
#ifndef BOOKLEAF_PACKAGES_SETUP_KERNEL_SHAPES_H
#define BOOKLEAF_PACKAGES_SETUP_KERNEL_SHAPES_H

#include <functional>

#include "common/constants.h"
#include "common/view.h"
#include "packages/setup/types.h"
#include "utilities/geometry/geometry.h"



namespace bookleaf {
namespace setup {
namespace kernel {

using constants::NCORN;

/** \brief Check if a point is inside a circle. */
bool
insideCircle(
        double const *shape_param,
        double const *point);

/** \brief Check if a point is inside a rectangle. */
bool
insideRectangle(
        double const *shape_param,
        double const *point);



/**
 * \brief Count the number of nodes inside the specified shape.
 *
 * \param [in] shape_param      specifies the shape in question
 * \param [in] x                nodal x-coordinates
 * \param [in] y                nodal y-coordinates
 * \param [in] is_inside_shape  function to check if a point is inside shape
 *
 * \returns number of nodes within the shape
 */
inline int
intersect(
        double const *shape_param,
        ConstView<double, NCORN> x,
        ConstView<double, NCORN> y,
        InsideFunc is_inside_shape)
{
    using constants::NCORN;

    int count = 0;
    for (int icn = 0; icn < NCORN; icn++) {
        double const point[2] = {x(icn), y(icn)};
        if (is_inside_shape(shape_param, point)) count++;
    }

    return count;
}



inline int
intersect(
        double const *shape_param,
        int iel,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        InsideFunc is_inside_shape)
{
    double _elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double _ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    ConstView<double, NCORN> elx(_elx);
    ConstView<double, NCORN> ely(_ely);

    return intersect(shape_param, elx, ely, is_inside_shape);
}



/**
 * \brief Approximate the volume fraction of the element specified by x and y
 *        that is contained within the shape specified by shape_param.
 *
 * \param [in] nit              current recursion iteration
 * \param [in] shape_param      specifies the shape in question
 * \param [in] x                nodal x-coordinates
 * \param [in] y                nodal y-coordinates
 * \param [in] is_inside_shape  function to check if a point is inside shape
 *
 * \returns volume fraction
 */
inline double
subdivide(
        int nit,
        double const *shape_param,
        ConstView<double, NCORN> x,
        ConstView<double, NCORN> y,
        InsideFunc is_inside_shape)
{
    using constants::NCORN;
    using constants::NDIM;
    int constexpr MAXITERATION = 10;

    double _centroid[NDIM];
    View<double, NDIM> centroid(_centroid);
    geometry::kernel::getCentroid(x, y, centroid);

    nit++;
    if (nit > MAXITERATION) {
        return is_inside_shape(shape_param, _centroid) ? 1.0 : 0.0;
    }

    double _x_loc[NCORN] = {0};
    double _y_loc[NCORN] = {0};
    View<double, NCORN> x_loc(_x_loc);
    View<double, NCORN> y_loc(_y_loc);

    //    y
    //    ^
    //  y2-      x----x
    //    |     /|  / |
    //    |    /  x   |
    //    |   / /   \ |
    //  y1-   x-------x
    //    |
    //    ----|-------|----> x
    //        x1      x2
    //

    // Recursively approximate the volume fraction by replacing a corner of
    // the element with the centroid, one at a time, and seeing if the resulting
    // element is full in or out of the parameterised shape.
    //
    double vf = 0.0;
    for (int i = 0; i < NCORN; i++) {
        x_loc(0) = x(i);
        y_loc(0) = y(i);
        int j = (i + 1) % NCORN;
        x_loc(1) = x(j);
        y_loc(1) = y(j);
        x_loc(2) = centroid(0);
        y_loc(2) = centroid(1);
        j = (i + 3) % NCORN;
        x_loc(3) = x(j);
        y_loc(3) = y(j);

        int const count = intersect(shape_param, x_loc, y_loc, is_inside_shape);
        if (count == NCORN) {
            vf += 1.0;

        } else if (count > 0 && count < NCORN) {
            vf += subdivide(nit, shape_param, x_loc, y_loc, is_inside_shape);
        }
    }

    return vf * 0.25;
}



inline double
subdivide(
        int nit,
        double const *shape_param,
        int iel,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        InsideFunc is_inside_shape)
{
    double _elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double _ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    ConstView<double, NCORN> elx(_elx);
    ConstView<double, NCORN> ely(_ely);

    return subdivide(nit, shape_param, elx, ely, is_inside_shape);
}

} // namepsace kernel
} // namepsace setup
} // namepsace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_KERNEL_SHAPES_H
