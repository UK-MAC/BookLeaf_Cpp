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
#include <iostream>
#include <cassert>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/timer_control.h"
#include "infrastructure/io/input_deck.h"
#include "common/error.h"
#include "common/config.h"
#include "common/runtime.h"
#include "common/sizes.h"
#include "common/timestep.h"
#include "common/data_control.h"
#include "common/cmd_args.h"

#include "packages/time/config.h"
#include "packages/hydro/config.h"
#include "packages/ale/config.h"
#include "utilities/data/global_configuration.h"
#include "utilities/eos/config.h"

#include "packages/setup/indicators.h"
#include "packages/setup/types.h"
#include "packages/setup/mesh_region.h"
#include "packages/setup/config.h"

#include "packages/check/driver/validate.h"

#include "utilities/io/config.h"
#include "utilities/comms/config.h"

#include "infrastructure/io/output_formatting.h"
#include "packages/io/driver/banner.h"
#include "infrastructure/io/read.h"
#include "infrastructure/io/write.h"
#include "infrastructure/init.h"
#include "infrastructure/solver.h"
#include "infrastructure/kill.h"



int
main(int argc, char *argv[])
{
    using namespace bookleaf;

#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Read command line arguments
    if (!INIT_CMD_ARGS(argc, argv)) {
        return EXIT_FAILURE;
    }

#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_MARK_BEGIN("initialisation");
#endif

    Config       config;
    Runtime      runtime;
    DataControl  data;
    TimerControl timers;
    Error        err;

    // -------------------------------------------------------------------------
    // Initialise parallelism
    // -------------------------------------------------------------------------
    inf::init::initParallelism(*config.comms, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    // -------------------------------------------------------------------------
    // Start timers
    // -------------------------------------------------------------------------
    timers.start(TimerID::TOTAL);
    timers.start(TimerID::INIT);

    // -------------------------------------------------------------------------
    // Print the welcome banner
    // -------------------------------------------------------------------------
    io::driver::banner(*config.comms->world);

    // -------------------------------------------------------------------------
    // Package configs need some pointers from main config
    // -------------------------------------------------------------------------
    config.time->comm = config.comms->spatial;
    config.time->io = config.io;

    config.hydro->comm = config.comms->spatial;
    config.hydro->global = config.global;
    config.hydro->io = config.io;
    config.hydro->eos = config.eos;

    config.ale->comm = config.comms->spatial;
    config.ale->global = config.global;
    config.ale->eos = config.eos;

    // -------------------------------------------------------------------------
    // Read input deck, rationalise and print
    // -------------------------------------------------------------------------
    inf::io::readInputDeck(CMD_ARGS.input_deck_file, config, runtime, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    inf::init::rationalise(config, runtime, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    inf::io::printConfiguration(CMD_ARGS.input_deck_file, config, runtime);

    // -------------------------------------------------------------------------
    // Initialise problem based on loaded configuration
    // -------------------------------------------------------------------------
    inf::init::init(config, runtime, timers, data, err);
    if (err.failed()) {
        halt(config, runtime, timers, data, err);
    }

    // Write initialisation information
    inf::io::writeOutput("initial_dump", config, runtime, timers, data);

    timers.stop(TimerID::INIT);

#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_MARK_END("initialisation");
#endif

    // -------------------------------------------------------------------------
    // Run solver
    // -------------------------------------------------------------------------
#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_init) data.dump("pre_solver");
#endif

    inf::control::solver(config, runtime, timers, data);

#if defined BOOKLEAF_DEBUG && defined BOOKLEAF_ZLIB_SUPPORT
    if (CMD_ARGS.dump_final) data.dump("post_solver");
#endif

    // Sync all data to host
    data.syncAllHost();

    // -------------------------------------------------------------------------
    // Finish
    // -------------------------------------------------------------------------
    // Write solution information
    inf::io::writeOutput("final_dump", config, runtime, timers, data);

    // Validate solution
    check::driver::validate(CMD_ARGS.input_deck_file, config, runtime, data, err);

    // Free memory
    inf::kill::kill(config);

    // Print timers
    timers.stop(TimerID::TOTAL);
    inf::io::printTimers(*config.comms, timers);

    // Print error message
    if (config.comms->world->zmproc) {
        if (err.failed()) {
            std::cout << inf::io::stripe() << "\n";
            std::cerr << err.serr << "\n";
        }

        std::cout << inf::io::stripe() << "\n";
    }

    // Clean up parallelism
    inf::kill::parallel(*config.comms);

    // Free command line arguments
    KILL_CMD_ARGS();

    return EXIT_SUCCESS;
}
