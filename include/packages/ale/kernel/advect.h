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
#ifndef BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECT_H
#define BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECT_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace ale {
namespace kernel {

using constants::NCORN;
using constants::NFACE;

void
updateBasisEl(
        double zerocut,
        double dencut,
        ConstDeviceView<double, VarDim> totv,
        ConstDeviceView<double, VarDim> totm,
        DeviceView<double, VarDim>      cutv,
        DeviceView<double, VarDim>      cutm,
        DeviceView<double, VarDim>      elvpr,
        DeviceView<double, VarDim>      elmpr,
        DeviceView<double, VarDim>      eldpr,
        DeviceView<double, VarDim>      elvolume,
        DeviceView<double, VarDim>      elmass,
        DeviceView<double, VarDim>      eldensity,
        int nel);

void
initBasisNd(
        DeviceView<double, VarDim> ndv0,
        DeviceView<double, VarDim> ndv1,
        DeviceView<double, VarDim> ndm0,
        int nnd);

void
calcBasisNd(
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim>           ndeln,
        ConstDeviceView<int, VarDim>           ndelf,
        ConstDeviceView<int, VarDim>           ndel,
        ConstDeviceView<double, VarDim>        elv0,
        ConstDeviceView<double, VarDim>        elv1,
        ConstDeviceView<double, VarDim, NCORN> cnm1,
        DeviceView<double, VarDim>             ndv0,
        DeviceView<double, VarDim>             ndv1,
        DeviceView<double, VarDim>             ndm0,
        int nnd);

void
fluxBasisNd(
        int id1,
        int id2,
        ConstDeviceView<int, VarDim, NFACE>    elel,
        ConstDeviceView<int, VarDim, NFACE>    elfc,
        ConstDeviceView<int, VarDim>           elsort,
        ConstDeviceView<double, VarDim, NFACE> fcdv,
        ConstDeviceView<double, VarDim, NFACE> fcdm,
        DeviceView<double, VarDim, NCORN>      cndv,
        DeviceView<double, VarDim, NCORN>      cndm,
        DeviceView<double, VarDim, NCORN>      cnflux,
        int nel);

void
massBasisNd(
        ConstDeviceView<int, VarDim, NCORN>    elnd,
        ConstDeviceView<int, VarDim>           ndeln,
        ConstDeviceView<int, VarDim>           ndelf,
        ConstDeviceView<int, VarDim>           ndel,
        ConstDeviceView<double, VarDim, NCORN> cnflux,
        DeviceView<double, VarDim, NCORN>      cnm1,
        DeviceView<double, VarDim>             ndm1,
        int nnd,
        int nel);

void
cutBasisNd(
        double cut,
        double dencut,
        ConstDeviceView<double, VarDim> ndv0,
        DeviceView<double, VarDim>      cutv,
        DeviceView<double, VarDim>      cutm,
        int nnd);

void
activeNd(
        int ibc,
        ConstDeviceView<int, VarDim>      ndstatus,
        ConstDeviceView<int, VarDim>      ndtype,
        DeviceView<unsigned char, VarDim> active,
        int nnd);

} // namespace kernel
} // namespace ale
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECT_H
