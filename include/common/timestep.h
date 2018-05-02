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
#ifndef BOOKLEAF_DATA_API_TIMESTEP_H
#define BOOKLEAF_DATA_API_TIMESTEP_H

#include <string>
#include <limits>

#include "common/error.h"
#include "packages/time/config.h"



namespace bookleaf {

struct Timestep
{
    int         nstep      = 0;             // Step # (first being 0)
    int         idtel      = 0;
    bool        zcorrector = false;
    std::string sdt        = " INITIAL";
    std::string mdt        = "   UNKNOWN";

    double time = -std::numeric_limits<double>::max();
    double dt = -std::numeric_limits<double>::max();
    double dts = -std::numeric_limits<double>::max();
};

inline void
rationalise(Timestep &timestep, time::Config const &time,
        Error &err __attribute__((unused)))
{
    timestep.time = time.time_start;
    timestep.dt = time.dt_initial;
}

} // namespace bookleaf



#endif // BOOKLEAF_DATA_API_TIMESTEP_H
