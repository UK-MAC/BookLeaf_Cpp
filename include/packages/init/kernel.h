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
#ifndef BOOKLEAF_PACKAGES_INIT_KERNEL_H
#define BOOKLEAF_PACKAGES_INIT_KERNEL_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {

struct Sizes;
class DataControl;

namespace init {
namespace kernel {

using constants::NCORN;
using constants::NFACE;

void
getElementConnectivity(
        ConstView<int, VarDim, NCORN> elnd,
        View<int, VarDim, NFACE>      elel,
        int nel);

void
getFaceConnectivity(
        ConstView<int, VarDim, NFACE> elel,
        View<int, VarDim, NFACE>      elfc,
        int nel);

void
correctConnectivity(
        int ncell,
        View<int, VarDim, NFACE> elel,
        View<int, VarDim, NFACE> elfc);

void
elMass(
        int nel,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elvolume,
        ConstView<double, VarDim, NCORN> cnwt,
        View<double, VarDim>             elmass,
        View<double, VarDim, NCORN>      cnmass);

void
mxMass(
        int ncp,
        ConstView<double, VarDim> mxdensity,
        ConstView<double, VarDim> mxvolume,
        View<double, VarDim>      mxmass);

void
nodeType(
        ConstView<int, VarDim, NCORN> elnd,
        View<int, VarDim>             ndtype,
        int nel,
        int nnd);

} // namespace kernel
} // namespace init
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_INIT_KERNEL_H
