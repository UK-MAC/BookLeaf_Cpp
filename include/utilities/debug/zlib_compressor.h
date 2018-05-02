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
#ifndef BOOKLEAF_UTILITIES_DEBUG_ZLIB_COMPRESSOR_H
#define BOOKLEAF_UTILITIES_DEBUG_ZLIB_COMPRESSOR_H

#ifndef BOOKLEAF_ZLIB_SUPPORT
static_assert(false, "BOOKLEAF_ZLIB_SUPPORT required");
#endif

#include <sstream>
#include <bitset>

#include <zlib.h>

#include "common/defs.h"



namespace bookleaf {

/**
 * \brief Incrementally add data to a compressed stream.
 */
class ZLibCompressor
{
public:
    typedef SizeType size_type;
    static size_type constexpr CHUNK = 262144;
    static int constexpr LEVEL = Z_BEST_COMPRESSION;

    explicit ZLibCompressor(std::ostream &_os);

    /** \brief Write bytes to the compressed output stream. */
    bool write(unsigned char const *src, size_type len);

    /** \brief Initialise the compressor. */
    bool init();

    /** \brief Finalise the output stream. */
    bool finish();

private:
    std::ostream &os;           //!< Output stream handle
    z_stream strm;              //!< Compressor state
    int zerr;                   //!< Compressor error
    unsigned char in [CHUNK];   //!< Input buffer
    unsigned char out[CHUNK];   //!< Output buffer
    size_type in_filled;        //!< Keep track of how full the input buffer is

    /** \brief Compress input buffer and write to output stream. */
    bool flush(int flush);
};

} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DEBUG_ZLIB_COMPRESSOR_H
