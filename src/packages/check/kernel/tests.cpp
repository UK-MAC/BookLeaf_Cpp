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
#include "packages/check/kernel/tests.h"

#include <cmath>
#include <limits>

#include "utilities/geometry/geometry.h"



namespace bookleaf {
namespace check {
namespace kernel {

using constants::NDIM;

namespace {

void
sodAnalytic(
        double xx,
        double time,
        View<double, 2> res)
{
    res(0) = 0.;
    res(1) = 0.;

    // Initialise
    res(0) = -std::numeric_limits<double>::max();
    res(1) = -std::numeric_limits<double>::max();

    // Calculate solution
    double const coeff = std::sqrt(1.4);

    if (xx <= -coeff * time) {
        res(0) = 1.0;
        res(1) = 2.5;

    } else if (xx <= -0.07027281256118326961 * time) {
        double const ss = xx/time;
        double const velocity = (5.0/6.0)*(ss+coeff);
        double const tt=std::pow(velocity-ss,2)/1.4;
        res(0) = std::pow(tt, 2.5);
        res(1) = res(0)*tt/(0.4*res(0));

    } else if (xx <= 0.92745262004894994908 * time) {
        res(0) = 0.42631942817849519385;
        res(1) = 1.77760006942335264781;

    } else {
        if (time <= 28.53627624872492075028) {
            if (xx <= 1.75215573203017816370 * time) {
                res(0) = 0.26557371170530706471;
                res(1) = 2.85354088799095973738;

            } else {
                res(0) = 0.125;
                res(1) = 2.0;
            }

        } else if (time < 40.68191725689148697880) {
            if (xx <= (50.0 - (time-28.53627624872492075028) * 1.01019363599101820444)) {
                res(0) = 0.26557371170530706471;
                res(1) = 2.85354088799095973738;

            } else {
                res(0) = 0.50939531774381657731;
                res(1) = 3.82996297061558546292;
            }
        }
    }
}

} // namespace

void
testSod(
        int nel,
        double time,
        ConstView<double, VarDim>        elvolume,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elenergy,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, 1>                  basis,
        View<double, 2>                  l1)
{
    double _centroid[NDIM];
    View<double, NDIM> centroid(_centroid, NDIM);

    double _solution[2];
    View<double, 2> solution(_solution, 2);

    // Initialise
    basis(0) = 0.;
    l1(0) = 0.;
    l1(1) = 0.;

    for (int iel = 0; iel < nel; iel++) {

        // Find centroid
        geometry::kernel::getCentroid(iel, cnx, cny, centroid);

        // Find solution
        sodAnalytic(centroid(0) - 50., time, solution);

        // Calculate components for L1 norm
        basis(0) += elvolume(iel);
        l1(0) += std::fabs(eldensity(iel) - solution(0)) * elvolume(iel);
        l1(1) += std::fabs(elenergy(iel) - solution(1)) * elvolume(iel);
    }
}

} // namespace kernel
} // namespace check
} // namespace bookleaf
