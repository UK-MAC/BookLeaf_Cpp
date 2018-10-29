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
#include "packages/hydro/driver/get.h"

#include <cassert>

#include "common/runtime.h"
#include "common/error.h"
#include "utilities/geometry/config.h"
#include "utilities/geometry/geometry.h"
#include "utilities/density/get_density.h"
#include "common/timer_control.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
getState(
        geometry::Config const &geom,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    // Update co-ordinates
    geometry::driver::getVertex(runtime, data);

    // Update geometry and iso-parametric terms
    geometry::driver::getGeometry(geom, *runtime.sizes, timers,
            TimerID::GETGEOMETRYL, data, err);
    if (err.failed()) return;

    // Update density
    density::driver::getDensity(*runtime.sizes, data);
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
