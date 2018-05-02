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
#ifndef BOOKLEAF_PACKAGES_HYDRO_KERNEL_INIT_H
#define BOOKLEAF_PACKAGES_HYDRO_KERNEL_INIT_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

using constants::NCORN;

void
initViscosity(
        int nel,
        View<double, VarDim>        elvisc,
        View<double, VarDim, NCORN> cnviscx,
        View<double, VarDim, NCORN> cnviscy);

void
initSubzonalPressureMass(
        int nel,
        ConstView<double, VarDim>        eldensity,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, VarDim, NCORN>      spmass);

} // namespace kernel
} // namespace hydro
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_HYDRO_DRIVER_INIT_H
