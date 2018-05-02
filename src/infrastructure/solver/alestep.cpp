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
#include "infrastructure/solver/alestep.h"

#include <iostream>

#include "packages/ale/driver/utils.h"
#include "packages/ale/driver/get_mesh_status.h"
#include "packages/ale/driver/get_flux_volume.h"
#include "packages/ale/driver/advect.h"
#include "infrastructure/solver/ale_update.h"
#include "common/timer_control.h"
#include "common/runtime.h"
#include "common/config.h"
#include "common/error.h"
#include "common/data_control.h"



namespace bookleaf {
namespace inf {
namespace solver {

void
alestep(
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data)
{
    // Determine if ALE is on
    if (ale::driver::isActive(*config.ale, *runtime.timestep)) {
        ScopedTimer st(timers, TimerID::ALESTEP);
        Error err;

        // Select mesh to be moved
        ale::driver::getMeshStatus(*config.ale, *runtime.sizes, timers,
                TimerID::ALEGETMESHSTATUS, data);

        // Calculate flux volume
        ale::driver::getFluxVolume(*config.ale, runtime, timers,
                TimerID::ALEGETFLUXVOLUME, data);

        // Advect independent variables
        ale::driver::advect(*config.ale, runtime, timers, data, err);
        if (err.failed()) {
            halt(config, runtime, timers, data, err);
        }

        // Update dependent variables
        inf::solver::aleUpdate(config, runtime, timers, data);
    }
}

} // namespace solver
} // namespace inf
} // namespace bookleaf
