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
#include "infrastructure/kill.h"

#include <cassert>

#include "common/config.h"
#include "common/error.h"
#include "common/data.h"

#include "utilities/comms/config.h"
#include "packages/hydro/config.h"
#include "utilities/eos/config.h"



namespace bookleaf {
namespace inf {
namespace kill {

void
parallel(
        comms::Comms &comms __attribute__((unused)))
{
    Kokkos::finalize();

    Error err;
    comms::killComms(err);
    if (err.failed()) {
        assert(false && "unhandled error");
        return;
    }
}



void
kill(
        Config &config)
{
    // Kill hydro config
    hydro::killHydroConfig(*config.hydro);

    // Kill EOS config
    killEOSConfig(*config.eos);

    // Kill partial sync
    Data::killPartialSync();
}

} // namespace kill
} // namespace inf
} // namespace bookleaf
