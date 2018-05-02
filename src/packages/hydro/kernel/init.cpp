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
#include "packages/hydro/kernel/init.h"

#include "common/constants.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
initViscosity(
        int nel,
        View<double, VarDim>        elvisc,
        View<double, VarDim, NCORN> cnviscx,
        View<double, VarDim, NCORN> cnviscy)
{
    // Zero the arrays
    for (int iel = 0; iel < nel; iel++) {
        elvisc(iel) = 0.;

        cnviscx(iel, 0) = 0.;
        cnviscx(iel, 1) = 0.;
        cnviscx(iel, 2) = 0.;
        cnviscx(iel, 3) = 0.;

        cnviscy(iel, 0) = 0.;
        cnviscy(iel, 1) = 0.;
        cnviscy(iel, 2) = 0.;
        cnviscy(iel, 3) = 0.;
    }
}



void
initSubzonalPressureMass(
        int nel,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, VarDim, NCORN>      spmass)
{
    for (int iel = 0; iel < nel; iel++) {
        double x3 = 0.25 * (cnx(iel, 0) + cnx(iel, 1) + cnx(iel, 2) + cnx(iel, 3));
        double y3 = 0.25 * (cny(iel, 0) + cny(iel, 1) + cny(iel, 2) + cny(iel, 3));

        for (int j1 = 0; j1 < NCORN; j1++) {
            double x1 = cnx(iel, j1);
            double y1 = cny(iel, j1);

            int j2 = (j1 + 1) % NCORN;
            double x2 = 0.5 * (x1 + cnx(iel, j2));
            double y2 = 0.5 * (y1 + cny(iel, j2));

            j2 = (j1 + 3) % NCORN;
            double x4 = 0.5 * (x1 + cnx(iel, j2));
            double y4 = 0.5 * (y1 + cny(iel, j2));

            double w1 = 0.25 * (-x1 + x2 + x3 - x4);
            double w2 = 0.25 * (-x1 - x2 + x3 + x4);
            double w3 = 0.25 * (-y1 + y2 + y3 - y4);
            double w4 = 0.25 * (-y1 - y2 + y3 + y4);

            spmass(iel, j1) = 4. * eldensity(iel) * (w1*w4 - w2*w3);
        }
    }
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
