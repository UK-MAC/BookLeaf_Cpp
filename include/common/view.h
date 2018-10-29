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
#ifndef BOOKLEAF_COMMON_VIEW_H
#define BOOKLEAF_COMMON_VIEW_H

#include <type_traits>
#include <cassert>

#include "common/defs.h"



namespace bookleaf {

// Views alias RAJA views

namespace internal {

template <
    typename T,
    SizeType NumRows,
    SizeType NumCols>
struct ViewWrapper
{
    typedef typename RAJA::View<T, RAJA::Layout<2>> type;

    static SizeType constexpr num_rows = NumRows;
    static SizeType constexpr num_cols = NumCols;
};

template <
    typename T,
    SizeType NumRows>
struct ViewWrapper<T, NumRows, 1>
{
    typedef typename RAJA::View<T, RAJA::Layout<1>> type;

    static SizeType constexpr num_rows = NumRows;
    static SizeType constexpr num_cols = 1;
};

} // namespace internal

template <
    typename T,
    SizeType NumRows,
    SizeType NumCols = 1>
using View = typename internal::ViewWrapper<T, NumRows, NumCols>::type;

template <
    typename T,
    SizeType NumRows,
    SizeType NumCols = 1>
using ConstView = typename internal::ViewWrapper<T const, NumRows, NumCols>::type;

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_VIEW_H
