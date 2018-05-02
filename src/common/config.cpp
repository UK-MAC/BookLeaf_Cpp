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
#include "common/config.h"

#include "packages/time/config.h"
#include "packages/hydro/config.h"
#include "packages/ale/config.h"
#include "packages/setup/config.h"

#include "utilities/eos/config.h"
#include "utilities/data/global_configuration.h"
#include "utilities/io/config.h"
#include "utilities/comms/config.h"



namespace bookleaf {

Config::Config() :
    time  (new time::Config()),
    hydro (new hydro::Config()),
    eos   (new EOS()),
    ale   (new ale::Config()),
    io    (new io_utils::Labels()),
    comms (new comms::Comms()),
    global(new GlobalConfiguration()),
    setup (new setup::Config())
{
}

} // namespace bookleaf
