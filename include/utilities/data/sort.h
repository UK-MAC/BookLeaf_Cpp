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
#ifndef BOOKLEAF_UTILITIES_DATA_SORT_H
#define BOOKLEAF_UTILITIES_DATA_SORT_H

#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
#include <memory>

#include "common/error.h"
#include "common/data_control.h"



namespace bookleaf {
namespace utils {
namespace kernel {

template <typename T, typename U>
void
sortIndices(
        ConstView<T, VarDim> keys,
        View<U, VarDim>    indices,
        int len,
        bool stable = false)
{
    // Initialise indices
    for (int i = 0; i < len; i++) indices(i) = i;

    // Compare indices based on corresponding key
    auto compare = [&keys](int i, int j) -> bool
    {
        return keys(i) < keys(j);
    };

    // Sort
    if (stable) {
        std::stable_sort(
                &indices(0),
                &indices(0) + len,
                compare);
    } else {
        std::sort(
                &indices(0),
                &indices(0) + len,
                compare);
    }
}



/**
 * @brief   Equivalent to Fortran's arr=arr(idx) syntax.
 */
template <typename T, typename U>
void
reorder(
        ConstView<T, VarDim> idx,
        View<U, VarDim>      arr,
        int len)
{
    // I don't think it's possible to do this without a temporary array, without
    // messing up the indices.
    std::unique_ptr<U[]> _scratch(new U[len]);
    View<U, VarDim> scratch(_scratch.get(), len);

    for (int i = 0; i < len; i++) {
        scratch(i) = arr(idx(i));
    }

    for (int i = 0; i < len; i++) {
        arr(i) = scratch(i);
    }
}

} // namespace kernel

namespace driver {

} // namespace driver
} // namespace utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DATA_SORT_H
