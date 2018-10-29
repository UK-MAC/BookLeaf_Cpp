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
#ifndef BOOKLEAF_UTILITIES_GEOMETRY_H
#define BOOKLEAF_UTILITIES_GEOMETRY_H

#include <vector>
#include <iostream>
#include <typeinfo>

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {

struct Sizes;
struct Runtime;
struct Error;

class TimerControl;
enum class TimerID : int;
class DataControl;

namespace geometry {
namespace driver {

void
getGeometry(
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data,
        Error &err);

void
getVertex(
        Runtime const &runtime,
        DataControl &data);

} // namespace driver

namespace kernel {

using constants::NDIM;
using constants::NCORN;
using constants::NFACE;

inline void
getCentroid(
        ConstView<double, NCORN> x,
        ConstView<double, NCORN> y,
        View<double, NDIM>       centroid)
{
    static_assert(NDIM == 2, "this routine only works for NDIM == 2");

    // Initialise
    centroid(0) = 0.;
    centroid(1) = 0.;

    // Calculate centroid
    double rvol = 0.;
    for (int i1 = 0; i1 < NCORN; i1++) {
        int const i2 = (i1 + 1) % NCORN;
        double const rr = x(i1)*y(i2) - x(i2)*y(i1);
        rvol += rr;
        centroid(0) += rr * (x(i1) + x(i2));
        centroid(1) += rr * (y(i1) + y(i2));
    }

    rvol = 0.5 * rvol;
    for (int i = 0; i < NDIM; i++) {
        centroid(i) = (rvol > 0.0) ? (centroid(i) / (6.0 * rvol)) : 0.0;
    }
}



inline void
getCentroid(
        int iel,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, NDIM>               centroid)
{
    double _elx[NCORN] = { cnx(iel, 0), cnx(iel, 1), cnx(iel, 2), cnx(iel, 3) };
    double _ely[NCORN] = { cny(iel, 0), cny(iel, 1), cny(iel, 2), cny(iel, 3) };

    ConstView<double, NCORN> elx(_elx);
    ConstView<double, NCORN> ely(_ely);

    getCentroid(elx, ely, centroid);
}



void
getIso(
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        DeviceView<double, VarDim>             a1,
        DeviceView<double, VarDim>             a2,
        DeviceView<double, VarDim>             a3,
        DeviceView<double, VarDim>             b1,
        DeviceView<double, VarDim>             b2,
        DeviceView<double, VarDim>             b3,
        DeviceView<double, VarDim, NCORN>      cnwt,
        int nel);

void
getVolume(
        ConstDeviceView<double, VarDim> a1,
        ConstDeviceView<double, VarDim> a3,
        ConstDeviceView<double, VarDim> b1,
        ConstDeviceView<double, VarDim> b3,
        DeviceView<double, VarDim>      volume,
        int len);

int
checkVolume(
        double val,
        ConstDeviceView<double, VarDim> volume,
        int nel);

void
getFluxVolume(
        double cut,
        ConstDeviceView<int, VarDim, NCORN> elnd,
        ConstDeviceView<double, VarDim>     ndx0,
        ConstDeviceView<double, VarDim>     ndy0,
        ConstDeviceView<double, VarDim>     ndx1,
        ConstDeviceView<double, VarDim>     ndy1,
        DeviceView<double, VarDim, NFACE>   fcdv,
        int nel);

void
getVertex(
        double dt,
        ConstDeviceView<double, VarDim> ndu,
        ConstDeviceView<double, VarDim> ndv,
        DeviceView<double, VarDim>      ndx,
        DeviceView<double, VarDim>      ndy,
        int nnd);

} // namespace kernel
} // namespace geometry
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_GEOMETRY_H
