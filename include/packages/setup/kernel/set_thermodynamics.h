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
#ifndef BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_THERMODYNAMICS_H
#define BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_THERMODYNAMICS_H

#include <vector>

#include "common/view.h"



namespace bookleaf {

struct Error;

namespace setup {

struct Config;
struct ThermodynamicsIC;

namespace kernel {

void
setThermodynamics(
        int nsize,
        ThermodynamicsIC const &tic,
        ConstView<int, VarDim>    flag,
        ConstView<double, VarDim> volume,
        View<double, VarDim>      density,
        View<double, VarDim>      energy);

void
rationaliseThermodynamics(
        setup::Config const &setup_config,
        int nmx,
        ConstView<double, VarDim> eldensity,
        ConstView<double, VarDim> elenergy,
        ConstView<int, VarDim>    mxfcp,
        ConstView<int, VarDim>    mxncp,
        ConstView<int, VarDim>    mxel,
        ConstView<int, VarDim>    cpmat,
        ConstView<int, VarDim>    cpnext,
        View<double, VarDim>      cpdensity,
        View<double, VarDim>      cpenergy,
        Error &err);

} // namespace kernel
} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_THERMODYNAMICS_H
