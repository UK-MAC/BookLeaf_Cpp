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
#include "infrastructure/solver/get.h"

#include "common/config.h"
#include "common/timer_control.h"
#include "packages/hydro/driver/get.h"



namespace bookleaf {
namespace inf {
namespace solver {

void
getEnergy(
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::GETENERGY);

    // XXX Missing code here that can't be merged

    // Hydro internal energy update
    hydro::driver::getEnergy(*config.hydro, runtime, data);

    // XXX Missing code here that can't be merged
}

} // namespace solver
} // namespace inf
} // namespace bookleaf
