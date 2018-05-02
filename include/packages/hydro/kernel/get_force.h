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
getForcePressure(
        ConstView<double, VarDim>   elpressure,
        ConstView<double, VarDim>   a1,
        ConstView<double, VarDim>   a3,
        ConstView<double, VarDim>   b1,
        ConstView<double, VarDim>   b3,
        View<double, VarDim, NCORN> cnfx,
        View<double, VarDim, NCORN> cnfy,
        int nel);

void
getForceViscosity(
        ConstView<double, VarDim, NCORN> edviscx,
        ConstView<double, VarDim, NCORN> edviscy,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel);

void
getForceSubzonalPressure(
        double const *pmeritreg,
        ConstView<int, VarDim>           elreg,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elcs2,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<double, VarDim, NCORN> spmass,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel);

void
getForceHourglass(
        double dt,
        double const *kappareg,
        ConstView<int, VarDim>           elreg,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elarea,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        View<double, VarDim, NCORN>      cnfx,
        View<double, VarDim, NCORN>      cnfy,
        int nel);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
