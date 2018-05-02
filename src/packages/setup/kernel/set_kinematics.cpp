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
#include "packages/setup/kernel/set_kinematics.h"

#include <cmath>

#include "packages/setup/types.h"
#include "utilities/geometry/geometry.h"



namespace bookleaf {
namespace setup {
namespace kernel {

void
setBackgroundKinematics(
        int nsize,
        KinematicsIC const &kic,
        ConstView<double, VarDim> xx,
        ConstView<double, VarDim> yy,
        View<double, VarDim>      uu,
        View<double, VarDim>      vv)
{
    switch (kic.geometry) {
    case KinematicsIC::Geometry::RADIAL:
        for (int i = 0; i < nsize; i++) {
            double const x = xx(i) - kic.params[1];
            double const y = yy(i) - kic.params[2];
            double rv = 1.0 / std::sqrt(x*x + y*y);
            rv *= kic.params[0];
            uu(i) = xx(i) * rv;
            vv(i) = yy(i) * rv;
        }
        break;

    case KinematicsIC::Geometry::PLANAR:
        for (int i = 0; i < nsize; i++) {
            uu(i) = kic.params[0];
            vv(i) = kic.params[1];
        }
        break;

    default:
        // Do nothing
        break;
    }
}



void
setRegionKinematics(
        int nel,
        int nnd __attribute__((unused)),
        KinematicsIC const &kic,
        ConstView<int, VarDim>           elreg,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<double, VarDim, NCORN> cnwt,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, VarDim>             uu,
        View<double, VarDim>             vv,
        View<double, VarDim>             wt)
{
    using constants::NDIM;

    double _point[NDIM];
    View<double, NDIM> point(_point);

    switch (kic.geometry) {
    case KinematicsIC::Geometry::RADIAL:
        for (int iel = 0; iel < nel; iel++) {
            if (elreg(iel) == kic.value) {
                geometry::kernel::getCentroid(
                        cnx.row(iel),
                        cny.row(iel),
                        point);

                double const x = point(0) - kic.params[1];
                double const y = point(1) - kic.params[2];
                double rv = 1.0 / std::sqrt(x*x + y*y);
                for (int ic = 0; ic < NCORN; ic++) {
                    int const ind = elnd(iel, ic);
                    uu(ind) += cnwt(iel, ic) * point(0) * rv * kic.params[0];
                    vv(ind) += cnwt(iel, ic) * point(1) * rv * kic.params[1];
                    wt(ind) += cnwt(iel, ic);
                }
            }
        }
        break;

    case KinematicsIC::Geometry::PLANAR:
        for (int iel = 0; iel < nel; iel++) {
            if (elreg(iel) == kic.value) {
                for (int ic = 0; ic < NCORN; ic++) {
                    int const ind = elnd(iel, ic);
                    uu(ind) += cnwt(iel, ic) * kic.params[0];
                    vv(ind) += cnwt(iel, ic) * kic.params[1];
                    wt(ind) += cnwt(iel, ic);
                }
            }
        }
        break;

    default:
        // Do nothing
        break;
    }
}



void
rationaliseKinematics(
        int nsize,
        double cutoff,
        ConstView<double, VarDim> wt,
        View<double, VarDim>      uu,
        View<double, VarDim>      vv)
{
    for (int ii = 0; ii < nsize; ii++) {
        if (std::abs(wt(ii)) < cutoff) {
            uu(ii) = 0.;
            vv(ii) = 0.;

        } else {
            double const rv = 1.0 / wt(ii);
            uu(ii) *= rv;
            vv(ii) *= rv;
        }
    }
}

} // namespace kernel
} // namespace setup
} // namespace bookleaf
