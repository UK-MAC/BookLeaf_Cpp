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
#ifndef BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_FLAGS_H
#define BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_FLAGS_H

#include <functional>
#include <vector>

#include "common/constants.h"
#include "common/view.h"
#include "packages/setup/types.h"



namespace bookleaf {

struct Sizes;
struct Error;
class DataControl;
enum class DataID : int;

namespace setup {

struct Shape;

namespace kernel {

using constants::NCORN;

namespace flag {

void
setCellFlags(
        int nsize,
        int itarget,
        int iindex,
        int nproc,
        ConstView<int, VarDim> test,
        View<int, VarDim>      flag,
        Error &err);

} // namespace flag

/** \brief Unconditionally set flags. */
void
setFlag(
        int nel,
        int iflag,
        View<int, VarDim> flag);

/** \brief Set flags where test matches a value. */
void
setFlagIf(
        int nel,
        int iflag,
        int itest,
        ConstView<int, VarDim> test,
        View<int, VarDim>      flag);



/** Handle shape flags individually for regions and materials. */

/** \brief Set region flags within a shape. */
void
setShapeRegion(
        int ireg,
        double const *shape_param,
        InsideFunc inside_shape,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                elreg,
        int nel);

/** \brief Set clean material flags within a shape. */
void
setShapeSingleMaterial(
        int itarget,
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                flag,
        int nel);

/** \brief Count new mixed-material elements within a shape. */
void
countShapeMixedMaterial(
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        ConstView<int, VarDim>           flag,
        int nel,
        int nmat,
        int &num_new_mixed_elements,
        int &num_new_components);

/** \brief Set new mixed-material element flags within a shape. */
void
setShapeMixedMaterial(
        int itarget,
        double const *param,
        InsideFunc inside,
        ConstView<double, VarDim, NCORN> cnx,
        ConstView<double, VarDim, NCORN> cny,
        View<int, VarDim>                elmat,
        View<int, VarDim>                mxel,
        View<int, VarDim>                mxncp,
        View<int, VarDim>                mxfcp,
        View<int, VarDim>                cpmat,
        View<int, VarDim>                cpprev,
        View<int, VarDim>                cpnext,
        View<double, VarDim>             frvolume,
        int nel,
        int nmat,
        int nmx,
        int ncp);

} // namespace kernel
} // namespace setup
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_SETUP_KERNEL_SET_FLAGS_H
