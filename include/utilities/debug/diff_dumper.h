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
#ifndef BOOKLEAF_UTILITIES_DEBUG_DIFF_DUMPER_H
#define BOOKLEAF_UTILITIES_DEBUG_DIFF_DUMPER_H

#include <string>
#include <fstream>
#include <iostream>
#include <type_traits>
#include <tuple>
#include <utility>

#include "common/view.h"

#ifdef BOOKLEAF_ZLIB_SUPPORT
#include "utilities/debug/zlib_compressor.h"
#endif

#include "utilities/debug/dump_utils.h"



namespace bookleaf {

#ifdef BOOKLEAF_ZLIB_SUPPORT

/**
 * \brief Small utility class that writes a set of views to 'pre' and 'post'
 *        files before and after execution of a block. This is used to generate
 *        unit test datasets.
 */
template <typename... Args>
class DiffDumper
{
private:
    template <typename... T>
    using refless_tuple =
            std::tuple<typename std::remove_reference<T>::type...>;

    typedef refless_tuple<Args...> tuple_type;
    typedef SizeType               size_type;

public:
    DiffDumper<Args...>(
            std::string _file,
            Args...     _views)
    :
        file(_file),
        views(_views...)
    {
        dump(true);
    }

    ~DiffDumper()
    {
        dump(false);
    }

private:
    std::string file;   //!< Base file name
    tuple_type  views;  //!< Tuple of views

    // Implement foreach over tuples
    template <SizeType I = 0, typename... Tp>
    inline typename std::enable_if<I == sizeof...(Tp), void>::type
    dumpViews(ZLibCompressor &, std::tuple<Tp...> &)
    {
        // Do nothing
    }

    template <SizeType I = 0, typename... Tp>
    inline typename std::enable_if<I < sizeof...(Tp), void>::type
    dumpViews(ZLibCompressor &zlc, std::tuple<Tp...>& t)
    {
        typedef typename std::tuple_element<I, tuple_type>::type view_type;
        view_type view = std::get<I>(t);

        // Only the order of the views matters, and since DataDump sorts them
        // alphabetically, label them a-z
        std::string name(1, (char) ('a' + I));

        writeZLibDump(zlc, name, view);
        dumpViews<I + 1, Tp...>(zlc, t);
    }

    void
    dump(bool pre)
    {
        std::string prefix = pre ? "pre_" : "post_";
        std::string path   = prefix + file + ".bldump";

        size_type num_views = std::tuple_size<tuple_type>::value;

        // Open stream and initialise compressor
        std::ofstream of(path.c_str(), std::ios_base::binary);
        ZLibCompressor zlc(of);
        zlc.init();

        // Write number of views
        zlc.write((unsigned char const *) &num_views, sizeof(size_type));

        // Write each view
        dumpViews(zlc, views);

        // Finalise stream
        zlc.finish();
        of.close();
    }
};



template <typename... Args>
DiffDumper<Args...>
makeDiffDumper(std::string file, Args... views)
{
    return DiffDumper<Args...>(file, views...);
}

#endif // BOOKLEAF_ZLIB_SUPPORT

} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DEBUG_DIFF_DUMPER_H
