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
#ifndef BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
#define BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

using constants::NCORN;

void
initAcceleration(
        View<double, VarDim> ndarea,
        View<double, VarDim> ndmass,
        View<double, VarDim> ndudot,
        View<double, VarDim> ndvdot,
        int nnd);

void
scatterAcceleration(
        double zerocut,
        ConstView<int, VarDim>           ndeln,
        ConstView<int, VarDim>           ndelf,
        ConstView<int, VarDim>           ndel,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim, NCORN> cnwt,
        ConstView<double, VarDim, NCORN> cnmass,
        ConstView<double, VarDim, NCORN> cnfx,
        ConstView<double, VarDim, NCORN> cnfy,
        View<double, VarDim>             ndarea,
        View<double, VarDim>             ndmass,
        View<double, VarDim>             ndudot,
        View<double, VarDim>             ndvdot,
        int nnd);

void
getAcceleration(
        double dencut,
        double zerocut,
        ConstView<double, VarDim> ndarea,
        View<double, VarDim>      ndmass,
        View<double, VarDim>      ndudot,
        View<double, VarDim>      ndvdot,
        int nnd);

void
applyAcceleration(
        double dt,
        View<double, VarDim> ndubar,
        View<double, VarDim> ndvbar,
        View<double, VarDim> ndu,
        View<double, VarDim> ndv,
        int nnd);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
