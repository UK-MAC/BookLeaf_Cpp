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
        ConstView<double, VarDim> totv,
        ConstView<double, VarDim> totm,
        View<double, VarDim>      cutv,
        View<double, VarDim>      cutm,
        View<double, VarDim>      elvpr,
        View<double, VarDim>      elmpr,
        View<double, VarDim>      eldpr,
        View<double, VarDim>      elvolume,
        View<double, VarDim>      elmass,
        View<double, VarDim>      eldensity,
        int nel);

void
initBasisNd(
        View<double, VarDim> ndv0,
        View<double, VarDim> ndv1,
        View<double, VarDim> ndm0,
        int nnd);

void
calcBasisNd(
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim>           elsort,
        ConstView<double, VarDim>        elv0,
        ConstView<double, VarDim>        elv1,
        ConstView<double, VarDim, NCORN> cnm1,
        View<double, VarDim>             ndv0,
        View<double, VarDim>             ndv1,
        View<double, VarDim>             ndm0,
        View<double, VarDim, NCORN>      cnm0,
        int nel);

void
fluxBasisNd(
        int id1,
        int id2,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<int, VarDim>           elsort,
        ConstView<double, VarDim, NFACE> fcdv,
        ConstView<double, VarDim, NFACE> fcdm,
        View<double, VarDim, NCORN>      cndv,
        View<double, VarDim, NCORN>      cndm,
        View<double, VarDim, NCORN>      cnflux,
        int nel);

void
massBasisNd(
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim>           elsort,
        ConstView<double, VarDim, NCORN> cnflux,
        View<double, VarDim, NCORN>      cnm1,
        View<double, VarDim>             ndm1,
        int nel);

void
cutBasisNd(
        double cut,
        double dencut,
        ConstView<double, VarDim> ndv0,
        View<double, VarDim>      cutv,
        View<double, VarDim>      cutm,
        int nnd);

void
activeNd(
        int ibc,
        ConstView<int, VarDim>      ndstatus,
        ConstView<int, VarDim>      ndtype,
        View<unsigned char, VarDim> active,
        int nnd);

} // namespace kernel
} // namespace ale
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECT_H
