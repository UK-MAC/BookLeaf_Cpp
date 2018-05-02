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
#ifndef BOOKLEAF_UTILITIES_IO_CONFIG_H
#define BOOKLEAF_UTILITIES_IO_CONFIG_H

#include <vector>
#include <string>
#include <iostream>

#include "common/sizes.h"
#include "utilities/misc/string_utils.h"



namespace bookleaf {
namespace io_utils {

// io_t: store human readable region/material labels
struct Labels
{
    std::vector<std::string> sregions = {"UNKNOWN"};
    std::vector<std::string> smaterials = {"UNKNOWN"};
};



inline void
rationalise(
        Labels &io,
        Sizes const &sizes,
        Error &err)
{
    // Helper macro to check if vector is large enough, and to shrink it
    // if it's too large
    #define CHECK_ARRAY_SIZE(V, N, ERR) { \
        if ((V).size() < (decltype((V).size())) (N)) { \
            err.fail((ERR)); \
            return; \
        } else if ((V).size() > (decltype((V).size())) (N)) { \
            (V).resize((N)); \
        } }

    CHECK_ARRAY_SIZE(io.sregions, sizes.nreg,
            "ERROR: inconsistent no. regions for IO");
    CHECK_ARRAY_SIZE(io.smaterials, sizes.nmat,
            "ERROR: inconsistent no. materials for IO");

    #undef CHECK_ARRAY_SIZE

    for (int i = 0; i < sizes.nreg; i++) {
        io.sregions[i].resize(10, ' ');
        if (trim(io.sregions[i]) == "UNKNOWN") {
            err.fail("ERROR: undefined region");
            return;
        }
    }

    for (int i = 0; i < sizes.nmat; i++) {
        io.smaterials[i].resize(10, ' ');
        if (trim(io.smaterials[i]) == "UNKNOWN") {
            err.fail("ERROR: undefined material");
            return;
        }
    }
}

} // namespace io_utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_IO_CONFIG_H
