#include "zlib_decompressor.h"

#include <cstring>
#include <cassert>



namespace bookleaf_diff {

ZLibDecompressor::ZLibDecompressor(std::istream &_is) : is(_is)
{
}



bool
ZLibDecompressor::read(unsigned char *dst, size_type len)
{
    // Partial read(s) from the end of the output buffer
    while (out_read + len > out_size) {

        if (out_read < out_size) {
            size_type const read = out_size - out_read;
            memcpy(dst, out + out_read, read);
            out_read += read;
            len -= read;
            dst += read;
        }

        // Get some more output to read
        if (!get_chunk()) return false;
    }

    // Full read from output buffer
    memcpy(dst, out + out_read, len);
    out_read += len;

    return true;
}



bool
ZLibDecompressor::init()
{
    zerr = Z_OK;
    in_size = 0;
    out_size = 0;
    out_read = 0;

    // Allocate inflate state
    strm.zalloc = Z_NULL;
    strm.zfree = Z_NULL;
    strm.opaque = Z_NULL;
    strm.avail_in = 0;
    strm.next_in = Z_NULL;

    zerr = inflateInit(&strm);
    return zerr == Z_OK;
}



void
ZLibDecompressor::finish()
{
    inflateEnd(&strm);
}



bool
ZLibDecompressor::get_chunk()
{
    // No data left to get
    if (is.eof() && in_size == 0) return false;

    // Read data until either the input buffer is full or there is nothing else
    // to read
    if (!is.eof()) {
        is.read((char *) in + in_size, CHUNK - in_size);
        in_size += is.gcount();
        assert(in_size <= CHUNK);
    }

    // Inflate some of the input
    strm.next_in   = in;
    strm.avail_in  = in_size;
    strm.next_out  = out;
    strm.avail_out = CHUNK;

    zerr = inflate(&strm, Z_NO_FLUSH);
    switch (zerr) {
    case Z_NEED_DICT:
        zerr = Z_DATA_ERROR; // fallthrough
    case Z_DATA_ERROR:
    case Z_MEM_ERROR:
        std::cerr << zerr << " error\n";
        return false;
    }

    if (zerr < 0) {
        std::cerr << zerr << " unexpected error\n";
        return false;
    }

    out_size = CHUNK - strm.avail_out;
    out_read = 0;

    if (zerr == Z_STREAM_END) {
        assert(strm.avail_in == 0);
    }

    // Move remaining input back to the start of the input buffer
    if (strm.avail_in > 0) {
        memmove(in, strm.next_in, strm.avail_in);
    }
    in_size = strm.avail_in;

    return true;
}

} // namespace bookleaf_diff
