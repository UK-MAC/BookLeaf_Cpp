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
#ifndef BOOKLEAF_UTILITIES_MIX_KERNEL_LIST_H
#define BOOKLEAF_UTILITIES_MIX_KERNEL_LIST_H

#include <algorithm>

#include "common/view.h"



namespace bookleaf {
namespace mix {
namespace kernel {

void
addEl(
        int iel,
        int imx,
        View<int, VarDim> elmat,
        View<int, VarDim> mxel,
        View<int, VarDim> mxncp,
        View<int, VarDim> mxfcp);

void
addCp(
        int imx,
        int icp,
        View<int, VarDim> mxfcp,
        View<int, VarDim> mxncp,
        View<int, VarDim> cpprev,
        View<int, VarDim> cpnext);

void
flattenIndex(
        int nmx,
        int ncp,
        ConstView<int, VarDim> sort,
        ConstView<int, VarDim> mxfcp,
        ConstView<int, VarDim> mxncp,
        ConstView<int, VarDim> cpmat,
        ConstView<int, VarDim> cpnext,
        View<int, VarDim>      cpprev,
        View<int, VarDim>      index);

void
flattenList(
        int nmx,
        int ncp,
        ConstView<int, VarDim> sort,
        View<int, VarDim>      mxfcp,
        View<int, VarDim>      mxel,
        View<int, VarDim>      mxncp,
        View<int, VarDim>      cpprev,
        View<int, VarDim>      cpnext);

template <typename T>
inline void
flattenQuant(
        int ncp,
        ConstView<int, VarDim> list,
        View<T, VarDim>        copy,
        View<T, VarDim>        quant)
{
    for (int i = 0; i < ncp; i++) {
        copy(i) = quant(i);
    }

    for (int i = 0; i < ncp; i++) {
        quant(i) = copy(list(i));
    }
}

} // namespace kernel
} // namespace mix
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_MIX_KERNEL_LIST_H
