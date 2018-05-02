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
#ifndef BOOKLEAF_PACKAGES_TIME_DRIVER_ADVANCE_H
#define BOOKLEAF_PACKAGES_TIME_DRIVER_ADVANCE_H



namespace bookleaf {

struct Sizes;
struct Timestep;
struct Runtime;
class TimerControl;
enum class TimerID : int;
struct Dt;
class DataControl;
struct Error;

namespace io_utils { struct Labels; }

namespace time {

struct Config;

namespace driver {

void calc(time::Config const &time, Timestep const &timestep, TimerID timerid,
        TimerControl &timers, Dt *&first, Dt *&current);

void end(time::Config const &time, Runtime &runtime, TimerControl &timers,
        DataControl &data, Dt *&current, Dt *&next, Error &err);

} // namespace driver
} // namespace time
} // namespace bookleaf



#endif // BOOKLEAF_PACKAGES_TIME_DRIVER_ADVANCE_H
