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

/**
 * \brief  Lightweight interface to 2 dimensional arrays.
 * \author timrlaw
 *
 * Uses std::enable_if throughout to leverage SFINAE, in order to selectively
 * include methods based on template arguments.
 */
template <
    typename T,
    SizeType NumRows,
    SizeType NumCols = 1>
class View
{
public:
    typedef          SizeType                   size_type;
    typedef          T                          value_type;
    typedef          T const                    const_value_type;
    typedef typename std::remove_const<T>::type nonconst_value_type;

    static_assert(NumCols != VarDim, "number of columns cannot be variable");

public:
    // -------------------------------------------------------------------------
    // Constructors
    // -------------------------------------------------------------------------
    /** \brief Fixed size constructor */
    template <SizeType _NumRows = NumRows>
    explicit View(
            typename std::enable_if<_NumRows != VarDim, T>::type *_ptr)
        : ptr(_ptr), num_rows(0) {}

    /** \brief Variable size constructor */
    template <SizeType _NumRows = NumRows>
    View(
            typename std::enable_if<_NumRows == VarDim, T>::type *_ptr,
            size_type _num_rows)
        : ptr(_ptr), num_rows(_num_rows) {}

    /** \brief Copy-constructor */
    View(View const &rhs) : ptr(rhs.ptr), num_rows(rhs.num_rows) {}

    /**
     * \brief Copy-constructor---non-const to const view
     *
     * Use SFINAE to only enable this copy-constructor when the current view is
     * const, otherwise it is a duplicate of the first copy-constructor and
     * the compiler will complain.
     */
    template <bool IsConst = std::is_const<value_type>::value>
    View(View<
            typename std::enable_if<IsConst, nonconst_value_type>::type,
            NumRows,
            NumCols
         > const &rhs)
        : ptr(rhs.ptr), num_rows(rhs.num_rows) {}

    friend View<const_value_type, NumRows, NumCols>;


    // -------------------------------------------------------------------------
    // Size accessors
    // -------------------------------------------------------------------------
    /** \brief Get the number of rows */
    BOOKLEAF_INLINE
    size_type
    rows() const { return NumRows == VarDim ? num_rows : NumRows; }

    /** \brief Get the number of columns */
    BOOKLEAF_INLINE static
    size_type constexpr
    cols() { return NumCols; }

    /** \brief Get the total size */
    BOOKLEAF_INLINE
    size_type
    size() const { return (NumRows == VarDim ? num_rows : NumRows) * NumCols; }


    // -------------------------------------------------------------------------
    // Data accessors
    // -------------------------------------------------------------------------
    /** \brief Return a raw data pointer */
    BOOKLEAF_INLINE
    value_type *
    data()
    {
        return ptr;
    }

    /** \brief Read/write 1D element accessor */
    template <SizeType _NumCols = NumCols>
    BOOKLEAF_INLINE
    typename std::enable_if<_NumCols == 1, value_type &>::type
    operator()(size_type i) const
    {
        assert(i < rows() && "invalid row index");

        return ptr[i];
    }

    /** \brief Read/write 2D element accessor */
    template <SizeType _NumCols = NumCols>
    BOOKLEAF_INLINE
    typename std::enable_if<_NumCols != 1, value_type &>::type
    operator()(size_type i, size_type j) const
    {
        assert(i < rows() && "invalid row index");
        assert(j < cols() && "invalid column index");

        return ptr[i * NumCols + j];
    }

    /** \brief Read/write flat accessor */
    BOOKLEAF_INLINE
    value_type &
    operator[](size_type idx) const
    {
        assert(idx < size() && "invalid index");

        return ptr[idx];
    }


private:
    T       * ptr;          //!< Data pointer
    size_type num_rows;     //!< Number of rows if NumRows == VarDim
};



/** \brief Alias for constant views */
template <
    typename T,
    SizeType NumRows,
    SizeType NumCols = 1>
using ConstView = View<T const, NumRows, NumCols>;

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_VIEW_H
