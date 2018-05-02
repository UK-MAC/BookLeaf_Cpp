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
#include "infrastructure/solver/get_dt.h"

#include <cassert>

#include "common/timer_control.h"
#include "common/dt.h"
#include "common/error.h"
#include "common/config.h"
#include "common/runtime.h"
#include "packages/time/driver/advance.h"
#include "packages/hydro/driver/get_dt.h"
#include "packages/ale/driver/get_dt.h"
#include "packages/ale/config.h"
#include "common/data_control.h"



namespace bookleaf {
namespace inf {
namespace solver {

void
getDt(
        Config const &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data)
{
    Error err;

    // To calculate the next delta-t, we construct a linked list of possible
    // values, and then search through this list for the minimum value.
    Dt *first, *current;
    first = current = nullptr;

    // ... Global timestep constraints
    time::driver::calc(*config.time, *runtime.timestep, TimerID::GETDT, timers,
            first, current);

    // ... Hydro timestep constraints
    hydro::driver::getDt(*config.hydro, *runtime.sizes, timers, data, current,
            err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    // ... ALE timestep constraints
    if (config.ale->zon) {
        ale::driver::getDt(*config.ale, *runtime.sizes, data, current);
    }

    // XXX Missing code here that can't be merged

    // ... Find minimum dt and set timestep
    time::driver::end(*config.time, runtime, timers, data, first, current, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }
}

} // namespace solver
} // namespace inf
} // namespace bookleaf
