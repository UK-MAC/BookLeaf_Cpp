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
#include "packages/time/config.h"

#include "common/error.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {
namespace time {

std::ostream &
operator<<(std::ostream &os, time::Config const &rhs)
{
    os << inf::io::format_value("Time at which calculation starts",
            "time_start", rhs.time_start);
    os << inf::io::format_value("Time at which calculation ends", "time_end",
            rhs.time_end);
    os << inf::io::format_value("Initial timestep", "dt_initial", rhs.dt_initial);
    os << inf::io::format_value("Minimum allowed timestep", "dt_min", rhs.dt_min);
    os << inf::io::format_value("Maximum allowed timestep", "dt_max", rhs.dt_max);
    os << inf::io::format_value("Timestep growth factor", "dt_g", rhs.dt_g);
    return os;
}



void
rationalise(time::Config const &time, Error &err)
{
    if (time.time_start > time.time_end) {
        err.fail("ERROR: time_start > time_end");
        return;
    }

    if (time.dt_g < 0.) {
        err.fail("ERROR: dt_g < 0");
        return;
    }

    if (time.dt_min < 0.) {
        err.fail("ERROR: dt_min < 0");
        return;
    }

    if (time.dt_max < 0.) {
        err.fail("ERROR: dt_max < 0");
        return;
    }

    if (time.dt_min > time.dt_max) {
        err.fail("ERROR: dt_min > dt_max");
        return;
    }

    if (time.dt_initial < 0.) {
        err.fail("ERROR: dt_initial < 0");
        return;
    }
}

} // namespace time
} // namespace bookleaf
