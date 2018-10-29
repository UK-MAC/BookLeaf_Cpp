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
#include "infrastructure/solver/lagstep.h"

#include <cassert>

#include "infrastructure/solver/get.h"

#include "common/timer_control.h"

#include "common/runtime.h"
#include "common/config.h"
#include "common/error.h"
#include "common/cmd_args.h"
#include "packages/time/driver/set.h"
#include "packages/hydro/driver/set.h"
#include "packages/hydro/driver/get.h"
#include "common/data_control.h"



namespace bookleaf {
namespace inf {
namespace solver {
namespace {

void
setPredictor(
        Runtime &runtime,
        DataControl &data)
{
    // Set timestep info
    time::driver::setPredictor(*runtime.timestep);

    // Set hydro
    hydro::driver::setPredictor(*runtime.sizes, data);
}



void
setCorrector(
        Runtime &runtime,
        DataControl &data)
{
    // Set timestep info
    time::driver::setCorrector(*runtime.timestep);

    // Set hydro
    hydro::driver::setCorrector(*runtime.sizes, data);
}



void
getState(
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data)
{
    // Update hydro state
    Error err;
    hydro::driver::getState(*config.geom, runtime, timers, data, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    // XXX Missing code here that can't be merged

    // Update internal energy
    inf::solver::getEnergy(config, runtime, timers, data);

    // XXX Missing code here that can't be merged

    // Update pressure and sound-speed
    hydro::driver::getEOS(*config.hydro, *runtime.sizes, timers, data);

    // XXX Missing code here that can't be merged
}

} // namespace

void
lagstep(
        Config const &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::LAGSTEP);

    // Predictor
    setPredictor(runtime, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_set_predictor");
#endif

    getForce(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_predictor_get_force");
#endif

    getState(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_predictor_get_state");
#endif

    // Corrector
    setCorrector(runtime, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_set_corrector");
#endif

    getForce(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_corrector_get_force");
#endif

    getAcceleration(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_get_acceleration");
#endif

    getState(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_lagstep) data.dump("lagstep_corrector_get_state");
#endif
}

} // namespace solver
} // namespace inf
} // namespace bookleaf
