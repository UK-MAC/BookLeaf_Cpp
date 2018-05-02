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
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NCORN> cnbasis,
        ConstView<double, VarDim, NFACE> fcdbasis,
        ConstView<double, VarDim>        elvar,
        View<double, VarDim, NFACE>      fcflux);

void
fluxNdVl(
        int ilsize,
        int iasize,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NCORN> cnbasis,
        ConstView<double, VarDim, NCORN> cndbasis,
        ConstView<double, VarDim, NCORN> cnvar,
        View<double, VarDim, NCORN>      cnflux);

void
updateEl(
        int id1,
        int id2,
        int ilsize,
        int iasize,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim>        elbase0,
        ConstView<double, VarDim>        elbase1,
        ConstView<double, VarDim>        cut,
        ConstView<double, VarDim, NFACE> fcflux,
        View<double, VarDim>             elflux,
        View<double, VarDim>             elvar);

void
updateNd(
        int iusize,
        int icsize,
        int insize,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<double, VarDim>        ndbase0,
        ConstView<double, VarDim>        ndbase1,
        ConstView<double, VarDim>        cut,
        ConstView<unsigned char, VarDim> active,
        ConstView<double, VarDim, NCORN> cnflux,
        View<double, VarDim>             ndflux,
        View<double, VarDim>             ndvar);

void
sumFlux(
        int id1,
        int id2,
        int ilsize,
        int iasize,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim, NFACE> fcflux,
        View<double, VarDim>             elflux);

} // namespace kernel
} // namespace ale
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_ALE_KERNEL_ADVECTORS_H
