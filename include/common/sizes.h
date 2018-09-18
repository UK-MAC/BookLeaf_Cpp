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
#ifndef BOOKLEAF_COMMON_SIZES_H
#define BOOKLEAF_COMMON_SIZES_H

#include "common/error.h"



namespace bookleaf {

struct Sizes
{
    int nel  = -1;  // Local mesh element #
    int nel1 = -1;  // nel + first ghost layer mesh element #
    int nel2 = -1;  // nel1 + second ghost layer mesh element #

    int nnd  = -1;  // Analagous to nel*
    int nnd1 = -1;  //      "
    int nnd2 = -1;  //      "

    int nmat = -1;  // Number of materials
    int nreg = -1;  // Number of regions

    // Multimaterial
    int ncp  =  0;
    int nmx  =  0;
    int mcp  =  0;
    int mmx  =  0;
};



inline void
rationalise(Sizes const &sizes, Error &err)
{
    if (sizes.nreg < 0) {
        err.fail("ERROR: incorrect value of nreg");
        return;
    }

    if (sizes.nmat < 0) {
        err.fail("ERROR: incorrect value of nmat");
        return;
    }
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_SIZES_H
