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
#ifndef BOOKLEAF_UTILITIES_DATA_COPY_H
#define BOOKLEAF_UTILITIES_DATA_COPY_H

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {
namespace utils {
namespace kernel {

using constants::NCORN;

template <typename T>
void
copy(
        View<T, VarDim>      dst,
        ConstView<T, VarDim> src,
        int len)
{
    for (int i = 0; i < len; i++) {
        dst(i) = src(i);
    }
}



template <typename T>
void
copy(
        View<T, VarDim, NCORN>      dst,
        ConstView<T, VarDim, NCORN> src,
        int len)
{
    for (int i = 0; i < len; i++) {
        for (int icn = 0; icn < NCORN; icn++) {
            dst(i, icn) = src(i, icn);
        }
    }
}

} // namespace kernel
} // namespace utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DATA_COPY_H
