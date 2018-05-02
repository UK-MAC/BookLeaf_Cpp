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
#include "infrastructure/solver.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/runtime.h"
#include "common/config.h"
#include "common/cmd_args.h"
#include "common/timestep.h"
#include "common/sizes.h"
#include "infrastructure/io/write.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "utilities/comms/config.h"
#include "infrastructure/solver/get_dt.h"
#include "packages/time/driver/advance.h"
#include "packages/time/driver/utils.h"

#include "infrastructure/solver/lagstep.h"
#include "infrastructure/solver/alestep.h"



namespace bookleaf {
namespace inf {
namespace control {

void
solver(
        Config const &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::SOLVER);

    // Time integration loop
    int iteration = 0;
    Timer tcycle;

#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_LOOP_BEGIN(mainloop, "solver loop");
    while (true) {
        CALI_CXX_MARK_LOOP_ITERATION(mainloop, iteration);
#else
    while (true) {
#endif

        timers.reset(tcycle);
        timers.start(tcycle);

        // Calculate timestep
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_getdt) data.dump("pre_getdt");
#endif
        inf::solver::getDt(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_getdt) data.dump("post_getdt");
#endif

        // XXX Missing code here that can't be merged

        // Lagrangian step
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_lagstep) data.dump("pre_lagstep");
#endif
        inf::solver::lagstep(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_lagstep) data.dump("post_lagstep");
#endif

        // ALE step
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_alestep) data.dump("pre_alestep");
#endif
        inf::solver::alestep(config, runtime, timers, data);
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
        if (CMD_ARGS.dump_alestep) data.dump("post_alestep");
#endif

        // XXX Missing code here that can't be merged

        timers.stop(tcycle);
        time::driver::printCycle(runtime.sizes->nel, *config.comms->world,
                *runtime.timestep, tcycle, TimerID::STEPIO, timers);

        // Test for end of calculation
        if (time::driver::finish(*runtime.timestep, *config.time)) break;

#ifdef BOOKLEAF_DEBUG
        if (CMD_ARGS.timestep_cap > 0) {
            if (runtime.timestep->nstep >= CMD_ARGS.timestep_cap) {
                break;
            }
        }
#endif

        // XXX Missing code here that can't be merged

        iteration++;
    }
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_LOOP_END(mainloop);
#endif
}

} // namespace control
} // namespace inf
} // namespace bookleaf
