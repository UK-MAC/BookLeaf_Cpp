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
#include "utilities/debug/zlib_compressor.h"

#include <cstring>
#include <cassert>



namespace bookleaf {

ZLibCompressor::ZLibCompressor(std::ostream &_os) :
    os(_os)
{
}



bool
ZLibCompressor::write(unsigned char const *src, size_type len)
{
    // Copy data to input buffer, and flush when it's full.
    size_type remaining = len;
    do {
        size_type const in_space = CHUNK - in_filled;
        size_type const copy_len = std::min(remaining, in_space);

        // Copy data from src to in buffer
        memcpy(in + in_filled, src, copy_len);
        src += copy_len;
        in_filled += copy_len;
        remaining -= copy_len;

        assert(in_filled <= CHUNK);
        assert(remaining <= len);

        // If there isn't enough data to fill a chunk yet, wait until the next
        // call to write() or finish()
        if (in_filled < CHUNK) return true;
        if (!flush(Z_NO_FLUSH)) return false;

    } while (remaining != 0);
    return true;
}



bool
ZLibCompressor::init()
{
    zerr = Z_OK;
    in_filled = 0;

    // Allocate deflate state
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;

    return (deflateInit(&strm, LEVEL) == Z_OK);
}



bool
ZLibCompressor::finish()
{
    if (in_filled > 0) {
        flush(Z_FINISH);

    } else {
        // TODO(timrlaw): Haven't tested this... requires data to exactly match
        // chunk size
        strm.next_in   = nullptr;
        strm.avail_in  = 0;
        strm.next_out  = nullptr;
        strm.avail_out = 0;

        zerr = deflate(&strm, Z_FINISH);
        if (zerr != Z_OK) {
            return false;
        }
    }

    deflateEnd(&strm);
    return true;
}



bool
ZLibCompressor::flush(int flush)
{
    strm.next_in  = in;
    strm.avail_in = in_filled;

    // Loop until deflate is finished
    do {
        strm.next_out  = out;
        strm.avail_out = CHUNK;

        // Deflate chunk
        zerr = deflate(&strm, flush);
        assert(zerr != Z_STREAM_ERROR);

        // Write to the output stream
        size_type const have = CHUNK - strm.avail_out;
        os.write((char const *) out, have);
        if (os.bad()) {
            return false;
        }

    } while (strm.avail_in > 0);

    // All input will be used
    assert(strm.avail_in == 0);
    in_filled = 0;
    return true;
}

} // namespace bookleaf
