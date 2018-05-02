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
#include "packages/time/driver/set.h"

#include "common/timestep.h"



namespace bookleaf {
namespace time {
namespace driver {

void
setPredictor(
        Timestep &timestep)
{
    timestep.zcorrector = false;
    timestep.dts = 0.5 * timestep.dt;
}



void
setCorrector(
        Timestep &timestep)
{
    timestep.zcorrector = true;
    timestep.dts = timestep.dt;
}

} // namespace driver
} // namespace time
} // namespace bookleaf
