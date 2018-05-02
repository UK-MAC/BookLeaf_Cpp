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
#ifndef BOOKLEAF_PACKAGES_TIME_CONFIG_H
#define BOOKLEAF_PACKAGES_TIME_CONFIG_H

#include <iostream>
#include <memory>



namespace bookleaf {

namespace comms { struct Comm; }
struct Error;

namespace io_utils { struct Labels; }

namespace time {

struct Config
{
    double time_start = 0.;
    double time_end   = 1.;
    double dt_g       = 1.02;
    double dt_min     = 1.e-8;
    double dt_max     = 1.e-1;
    double dt_initial = 1.e-5;

    std::shared_ptr<comms::Comm>      comm;
    std::shared_ptr<io_utils::Labels> io;
};



std::ostream &
operator<<(
        std::ostream &os,
        time::Config const &rhs);

void
rationalise(
        time::Config const &time,
        Error &err);

} // namespace time
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_TIME_CONFIG_H
