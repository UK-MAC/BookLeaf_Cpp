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
#ifndef BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECTORS_H
#define BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECTORS_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace ale {
namespace kernel {

using constants::NCORN;
using constants::NFACE;

void
fluxElVl(
        int id1,
        int id2,
        int ilsize,
        int iasize,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim, NCORN> cnbasis,
        ConstDeviceView<double, VarDim, NFACE> fcdbasis,
        ConstDeviceView<double, VarDim>        elvar,
        DeviceView<double, VarDim, NFACE>      fcflux);

void
fluxNdVl(
        int ilsize,
        int iasize,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim, NCORN> cnbasis,
        ConstDeviceView<double, VarDim, NCORN> cndbasis,
        ConstDeviceView<double, VarDim, NCORN> cnvar,
        DeviceView<double, VarDim, NCORN>      cnflux);

void
updateEl(
        int id1,
        int id2,
        int ilsize,
        int iasize,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim>        elbase0,
        ConstDeviceView<double, VarDim>        elbase1,
        ConstDeviceView<double, VarDim>        cut,
        ConstDeviceView<double, VarDim, NFACE> fcflux,
        DeviceView<double, VarDim>             elflux,
        DeviceView<double, VarDim>             elvar);

void
updateNd(
        int iusize,
        int icsize,
        int insize,
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim>           ndeln,
        ConstDeviceView<int, VarDim>           ndelf,
        ConstDeviceView<int, VarDim>           ndel,
        ConstDeviceView<double, VarDim>        ndbase0,
        ConstDeviceView<double, VarDim>        ndbase1,
        ConstDeviceView<double, VarDim>        cut,
        ConstDeviceView<unsigned char, VarDim> active,
        ConstDeviceView<double, VarDim, NCORN> cnflux,
        DeviceView<double, VarDim>             ndflux,
        DeviceView<double, VarDim>             ndvar);

void
sumFlux(
        int id1,
        int id2,
        int ilsize,
        int iasize,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<double, VarDim, NFACE> fcflux,
        DeviceView<double, VarDim>             elflux);

} // namespace kernel
} // namespace ale
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECTORS_H
