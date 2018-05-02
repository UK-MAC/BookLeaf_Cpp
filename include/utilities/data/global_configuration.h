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
#ifndef BOOKLEAF_UTILITIES_DATA_GLOBAL_CONFIGURATION_H
#define BOOKLEAF_UTILITIES_DATA_GLOBAL_CONFIGURATION_H

#include <iostream>

#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {

/** \brief Store configurable values used throughout the code. */
struct GlobalConfiguration
{
    double zcut    = 1.e-8;
    double zerocut = 1.e-40;
    double dencut  = 1.e-6;
    double accut   = 1.e-6;
};



inline std::ostream &
operator<<(
        std::ostream &os,
        GlobalConfiguration const &rhs)
{
    os << inf::io::format_value("Rounding precision cut-off", "zcut", rhs.zcut);
    os << inf::io::format_value("Underflow cut-off", "zerocut", rhs.zerocut);
    os << inf::io::format_value("Density cut-off", "dencut", rhs.dencut);
    os << inf::io::format_value("Acceleration cut-off", "accut", rhs.accut);
    return os;
}



inline void
rationalise(
        GlobalConfiguration const &global,
        Error &err)
{
    if (global.zcut < 0.) {
        err.fail("ERROR: zcut < 0");
        return;
    }

    if (global.zerocut < 0.) {
        err.fail("ERROR: zerocut < 0");
        return;
    }

    if (global.dencut < 0.) {
        err.fail("ERROR: dencut < 0");
        return;
    }

    if (global.accut < 0.) {
        err.fail("ERROR: accut < 0");
        return;
    }
}

} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DATA_GLOBAL_CONFIGURATION_H
