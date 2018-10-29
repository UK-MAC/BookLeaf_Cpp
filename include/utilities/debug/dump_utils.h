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
#ifndef BOOKLEAF_UTILITIES_DEBUG_DUMP_UTILS_H
#define BOOKLEAF_UTILITIES_DEBUG_DUMP_UTILS_H

#include "common/defs.h"
#include "common/view.h"

#ifdef BOOKLEAF_ZLIB_SUPPORT
#include "utilities/debug/zlib_compressor.h"
#endif



namespace bookleaf {

#ifdef BOOKLEAF_ZLIB_SUPPORT
/**
 * \brief   Serialise a data item to a compressed output stream.
 */
template <
    typename T,
    SizeType NumRows,
    SizeType NumCols = 1>
void
writeZLibDump(
        ZLibCompressor &zlc,
        std::string const name,
        View<T, NumRows, NumCols> view)
{
    SizeType const name_len = name.size();
    SizeType const size     = view.size();

    // Get type name
    typedef View<T, NumRows, NumCols> view_type;
    typedef typename view_type::value_type value_type;
    std::string const type_name = getTypeName<value_type>();
    SizeType const type_name_len = type_name.size();

    // Write name
    zlc.write((unsigned char const *) &name_len, sizeof(SizeType));
    zlc.write((unsigned char const *) name.c_str(), name_len);

    // Write size
    zlc.write((unsigned char const *) &size, sizeof(SizeType));

    if (size == 0) return;

    // Write type name
    zlc.write((unsigned char const *) &type_name_len, sizeof(SizeType));
    zlc.write((unsigned char const *) type_name.c_str(), type_name_len);

    // Write data
    zlc.write((unsigned char const *) &view[0], size * sizeof(value_type));
}
#endif // BOOKLEAF_ZLIB_SUPPORT

} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DEBUG_DUMP_UTILS_H
