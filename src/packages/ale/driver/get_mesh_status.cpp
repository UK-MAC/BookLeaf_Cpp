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
#include "packages/ale/driver/get_mesh_status.h"

#include "common/sizes.h"
#include "common/timer_control.h"
#include "packages/ale/config.h"
#include "common/data_control.h"
#include "packages/ale/kernel/get_mesh_status.h"



namespace bookleaf {
namespace ale {
namespace driver {

void
getMeshStatus(ale::Config const &ale, Sizes const &sizes, TimerControl &timers,
        TimerID timerid, DataControl &data)
{
    ScopedTimer st(timers, timerid);

    auto ndstatus = data[DataID::ALE_INDSTATUS].host<int, VarDim>();

    // Select mesh to be moved
    kernel::getMeshStatus(sizes.nnd, ale.zeul, ndstatus);
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
