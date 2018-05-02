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
#ifndef BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_KINEMATICS_H
#define BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_KINEMATICS_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {

struct Sizes;

namespace setup {

struct KinematicsIC;

namespace kernel {

using constants::NCORN;

void
setBackgroundKinematics(
        int nsize,
        KinematicsIC const &kic,
        ConstView<double, VarDim> xx,
        ConstView<double, VarDim> yy,
        View<double, VarDim>      uu,
        View<double, VarDim>      vv);

void
setRegionKinematics(
        int nel,
        int nnd __attribute__((unused)),
        KinematicsIC const &kic,
        ConstView<int, VarDim>           elreg,
        ConstView<int, VarDim, NCORN>    elnd,
        ConstView<double, VarDim, NCORN> cnwt,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<double, VarDim>             uu,
        View<double, VarDim>             vv,
        View<double, VarDim>             wt);

void
rationaliseKinematics(
        int nsize,
        double cutoff,
        ConstView<double, VarDim> wt,
        View<double, VarDim>      uu,
        View<double, VarDim>      vv);

} // namespace kernel
} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_KINEMATICS_H
