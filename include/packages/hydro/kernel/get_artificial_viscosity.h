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
using constants::NFACE;

void
initArtificialViscosity(
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim>             elvisc,
        DeviceView<double, VarDim, NFACE>      dx,
        DeviceView<double, VarDim, NFACE>      dy,
        DeviceView<double, VarDim, NFACE>      du,
        DeviceView<double, VarDim, NFACE>      dv,
        DeviceView<double, VarDim, NCORN>      cnviscx,
        DeviceView<double, VarDim, NCORN>      cnviscy,
        int nel);

void
limitArtificialViscosity(
        int nel,
        double zerocut,
        double cvisc1,
        double cvisc2,
        ConstDeviceView<int, VarDim>           ndtype,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim>        eldensity,
        ConstDeviceView<double, VarDim>        elcs2,
        ConstDeviceView<double, VarDim, NFACE> du,
        ConstDeviceView<double, VarDim, NFACE> dv,
        ConstDeviceView<double, VarDim, NFACE> dx,
        ConstDeviceView<double, VarDim, NFACE> dy,
        DeviceView<double, VarDim, NCORN>      scratch,
        DeviceView<double, VarDim, NFACE>      cnviscx,
        DeviceView<double, VarDim, NFACE>      cnviscy,
        DeviceView<double, VarDim>             elvisc);

void
getArtificialViscosity(
        double zerocut,
        ConstDeviceView<double, VarDim, NCORN> cnx,
        ConstDeviceView<double, VarDim, NCORN> cny,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        DeviceView<double, VarDim, NFACE>      cnviscx,
        DeviceView<double, VarDim, NFACE>      cnviscy,
        int nel);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
