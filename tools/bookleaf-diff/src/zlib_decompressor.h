#ifndef BOOKLEAF_DIFF_ZLIB_DECOMPRESSOR_H
#define BOOKLEAF_DIFF_ZLIB_DECOMPRESSOR_H

#include <iostream>

#ifndef BOOKLEAF_ZLIB_SUPPORT
static_assert(false, "BOOKLEAF_ZLIB_SUPPORT required");
#endif

#include <zlib.h>



namespace bookleaf_diff {

/**
 * \brief Read a compressed dump.
 */
class ZLibDecompressor
{
public:
    typedef std::size_t size_type;
    static size_type constexpr CHUNK = 262144;

    explicit ZLibDecompressor(std::istream &_is);

    /** \brief Read bytes from the compressed input stream. */
    bool read(unsigned char *dst, size_type len);

    /** \brief Initialise the decompressor. */
    bool init();

    /** \brief Shutdown the decompressor. */
    void finish();

private:
    std::istream &is;           //!< Output stream handle
    z_stream strm;              //!< Compressor state
    int zerr;                   //!< Compressor error
    unsigned char in [CHUNK];   //!< Input buffer
    unsigned char out[CHUNK];   //!< Output buffer
    size_type in_size;          //!< Num bytes in input buffer
    size_type out_size;         //!< Num bytes in output buffer
    size_type out_read;         //!< Num bytes read from output buffer

    /** \brief Read & decompress a chunk into the output buffer. */
    bool get_chunk();
};

} // namespace bookleaf_diff



#endif // BOOKLEAF_DIFF_ZLIB_DECOMPRESSOR_H
