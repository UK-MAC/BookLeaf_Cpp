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
#ifndef BOOKLEAF_PACKAGES_TIME_DRIVER_UTILS_H
#define BOOKLEAF_PACKAGES_TIME_DRIVER_UTILS_H



namespace bookleaf {

namespace comms { struct Comm; }
struct Timestep;
struct Timer;
class TimerControl;
enum class TimerID : int;

namespace time {

struct Config;

namespace driver {

void
printCycle(
        int nel,
        comms::Comm const &comm,
        Timestep const &timestep,
        Timer tcycle,
        TimerID timerid,
        TimerControl &timers);

bool
finish(
        Timestep const &timestep,
        time::Config const &time);

} // namespace driver
} // namespace time
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_TIME_DRIVER_UTILS_H
