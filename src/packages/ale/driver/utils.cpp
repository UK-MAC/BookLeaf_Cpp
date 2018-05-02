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
#include "packages/ale/driver/utils.h"

#include "packages/ale/config.h"
#include "common/timestep.h"



namespace bookleaf {
namespace ale {
namespace driver {

bool
isActive(
        ale::Config const &ale,
        Timestep const &timestep)
{
    // Check ALE requested
    if (!ale.zexist) {
        return false;
    }

    // Check ALE currently on
    return (timestep.time >= ale.mintime) &&
           (timestep.time <= ale.maxtime);
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
