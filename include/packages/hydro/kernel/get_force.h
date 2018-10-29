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
        ConstDeviceView<double, VarDim>   elpressure,
        ConstDeviceView<double, VarDim>   a1,
        ConstDeviceView<double, VarDim>   a3,
        ConstDeviceView<double, VarDim>   b1,
        ConstDeviceView<double, VarDim>   b3,
        DeviceView<double, VarDim, NCORN> cnfx,
        DeviceView<double, VarDim, NCORN> cnfy,
        int nel);

void
getForceViscosity(
        ConstDeviceView<double, VarDim, NCORN> edviscx,
        ConstDeviceView<double, VarDim, NCORN> edviscy,
        DeviceView<double, VarDim, NCORN>      cnfx,
        DeviceView<double, VarDim, NCORN>      cnfy,
        int nel);

void
getForceSubzonalPressure(
        ConstDeviceView<double, VarDim>        pmeritreg,
        ConstDeviceView<int, VarDim>           elreg,
        ConstDeviceView<double, VarDim>        eldensity,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        ConstDeviceView<double, VarDim, NCORN> spmass,
        DeviceView<double, VarDim, NCORN>      cnfx,
        DeviceView<double, VarDim, NCORN>      cnfy,
        int nel);

void
getForceHourglass(
        double dt,
        ConstDeviceView<double, VarDim>        kappareg,
        ConstDeviceView<int, VarDim>           elreg,
        ConstDeviceView<double, VarDim>        eldensity,
        ConstDeviceView<double, VarDim>        elarea,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim, NCORN>      cnfx,
        DeviceView<double, VarDim, NCORN>      cnfy,
        int nel);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
