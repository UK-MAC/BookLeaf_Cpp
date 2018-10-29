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
getEnergy(
        double dt,
        double zerocut,
        ConstDeviceView<double, VarDim, NCORN> cnfx,
        ConstDeviceView<double, VarDim, NCORN> cnfy,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        ConstDeviceView<double, VarDim>        elmass,
        DeviceView<double, VarDim>             elenergy,
        int nel);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
