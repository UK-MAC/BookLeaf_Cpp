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
#include "packages/io/driver/banner.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <stdexcept>

#include "common/error.h" // TOSTRING
#include "utilities/comms/config.h"



/*Far too complicated banner printing
  Easy to add authors though*/
namespace bookleaf {
namespace io {
namespace driver {
namespace {

/*print a string inside the banner starting at start*/
void
printLine(size_t len, size_t start, std::string str)
{
    if (start > len + str.size() || start < 2) {
        assert(false && "trying to print outside of banner");
    }

    std::cout
        << std::string(start - 1, ' ')
        << str
        << std::string(len + 3 - (size_t) str.size() - start, ' ')
        << std::endl;
}

/* Base case for longestString */
size_t
longestString()
{
    return 0;
}

/* Can take any number of std::strings and returns length of the longest */
template <typename... Ts>
size_t
longestString(std::string head, const Ts&... tail)
{
    return std::max(head.size(), longestString(tail...));
}

/*base case for print authors*/
void
printAuthors(size_t len __attribute__((unused)),
        int start __attribute__((unused)))
{
}

/*prints any number of additional authors under the first*/
template <typename... Ts>
void
printAuthors(size_t len, int start, std::string head, const Ts&... tail)
{
    printLine(len, start, head);
    printAuthors(len, start, tail...);
}

/*prints a banner last paramters can be any number of authors*/
template <typename T, typename... Ts>
void
printBanner(char border, std::string proginfo, std::string revision,
        const T &head, const Ts&... tail)
{
    // Make sure these three are the same length including white space so
    // everything lines up
    std::string const progname       = "BookLeaf:  ";
    std::string const author_intro   = "Author(s): ";
    std::string const revision_intro = "Revision:  ";

    size_t longest = longestString(progname + proginfo,
                       revision_intro + revision,
                       author_intro + head,
                       tail...);

    const std::string full_line = std::string(1, ' ') + std::string(131, border);

    std::cout << full_line << std::endl;
    printLine(longest,2,progname + proginfo);
    std::cout << std::endl;
    printLine(longest,2, author_intro + head);
    printAuthors(longest,author_intro.size() + 2, tail...);
    std::cout << std::endl;
    printLine(longest,2,revision_intro + revision);
    std::cout << full_line << std::endl;
}

} // namespace

void
banner(
        comms::Comm const &comm)
{
    if (!comm.zmproc) return;

    std::string version = "Unversioned directory";
#if defined BOOKLEAF_GIT_BRANCH && defined BOOKLEAF_GIT_COMMIT_HASH
    version = std::string(TOSTRING(BOOKLEAF_GIT_BRANCH)) +
              "@" +
              std::string(TOSTRING(BOOKLEAF_GIT_COMMIT_HASH));
#endif

    printBanner('#',
            "Light-weight FE Hydro scheme",
            version,
            "R. Kevis",
            "D. Harris");
}

} // namespace driver
} // namespace io
} // namespace bookleaf
