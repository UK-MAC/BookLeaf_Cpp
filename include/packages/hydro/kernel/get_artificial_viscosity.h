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
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        View<double, VarDim>             elvisc,
        View<double, VarDim, NFACE>      dx,
        View<double, VarDim, NFACE>      dy,
        View<double, VarDim, NFACE>      du,
        View<double, VarDim, NFACE>      dv,
        View<double, VarDim, NCORN>      cnviscx,
        View<double, VarDim, NCORN>      cnviscy,
        int nel);

void
limitArtificialViscosity(
        int nel,
        double zerocut,
        double cvisc1,
        double cvisc2,
        ConstView<int, VarDim>           ndtype,
        ConstView<int, VarDim, NFACE>    elel,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<int, VarDim, NFACE>    elfc,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim>        elcs2,
        ConstView<double, VarDim, NFACE> du,
        ConstView<double, VarDim, NFACE> dv,
        ConstView<double, VarDim, NFACE> dx,
        ConstView<double, VarDim, NFACE> dy,
        View<double, VarDim, NCORN>      scratch,
        View<double, VarDim, NFACE>      cnviscx,
        View<double, VarDim, NFACE>      cnviscy,
        View<double, VarDim>             elvisc);

void
getArtificialViscosity(
        double zerocut,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<double, VarDim, NCORN> cnu,
        ConstView<double, VarDim, NCORN> cnv,
        View<double, VarDim, NFACE>      cnviscx,
        View<double, VarDim, NFACE>      cnviscy,
        int nel);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_KERNEL_GET_H
