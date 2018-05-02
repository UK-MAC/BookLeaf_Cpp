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
#include "common/error.h"

#include <iostream>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/config.h"
#include "utilities/comms/config.h"
#include "infrastructure/io/write.h"
#include "common/timer_control.h"
#include "infrastructure/io/output_formatting.h"



namespace bookleaf {

void
halt(
        Config const &config,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data,
        Error err)
{
    // Whether to print from this rank
    bool const zout = ((err.iout == Error::ErrorHalt::HALT_ALL) &&
                       (config.comms->world->zmproc)) ||
                      (err.iout == Error::ErrorHalt::HALT_SINGLE);

    if (zout) {
        inf::io::stripe();
        std::cerr << inf::io::stripe() << "\n";
    }

    // Output
    if (err.iout == Error::ErrorHalt::HALT_ALL) {
        inf::io::writeOutput("final_dump", config, runtime, timers, data);
    }

    // Halt timers
    timers.stopAll();

    // Print timers
    if (zout) {
        std::cout << timers << "\n";
    }

    // End timers
    if (err.failed()) {
        if (zout) {
            std::cerr << err.serr << "\n";
        }
    }

    // Spacer
    if (zout) {
        std::cerr << inf::io::stripe() << "\n";
    }

#ifdef BOOKLEAF_MPI_SUPPORT
    if (err.iout == Error::ErrorHalt::HALT_ALL) {
        TYPH_Kill(true);
    } else {
        TYPH_Abort(-1);
    }
#endif

    // XXX Should not get here if MPI is enabled
    std::exit(err.failed() ? EXIT_FAILURE : EXIT_SUCCESS);
}

} // namespace bookleaf
