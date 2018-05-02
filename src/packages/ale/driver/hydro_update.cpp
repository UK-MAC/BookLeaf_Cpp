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
#include "packages/ale/driver/hydro_update.h"

#include "packages/ale/config.h"
#include "utilities/geometry/geometry.h"
#include "utilities/density/get_density.h"
#include "utilities/eos/get_eos.h"
#include "common/timer_control.h"



namespace bookleaf {
namespace ale {
namespace driver {

void
hydroUpdate(
        ale::Config const &ale,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    // update geometry
    geometry::driver::getGeometry(sizes, timers, TimerID::GETGEOMETRYA, data,
            err);

    // update density to be consistent with geometry
    density::driver::getDensity(sizes, data);

    // update EoS
    eos::driver::getEOS(*ale.eos, sizes, timers, TimerID::GETEOSA, data);
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
