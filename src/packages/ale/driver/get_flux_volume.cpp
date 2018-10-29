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
#include "packages/ale/driver/get_flux_volume.h"

#include "common/runtime.h"
#include "common/sizes.h"
#include "common/timestep.h"
#include "packages/ale/config.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "utilities/data/global_configuration.h"
#include "utilities/geometry/geometry.h"
#include "utilities/data/copy.h"
#include "packages/ale/kernel/get_mesh_velocity.h"
#include "common/constants.h"



namespace bookleaf {
namespace ale {
namespace driver {

void
getFluxVolume(
        ale::Config const &ale,
        Runtime const &runtime,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, timerid);

    int const nel = runtime.sizes->nel;
    int const nnd = runtime.sizes->nnd;

    auto elnd   = data[DataID::IELND].cdevice<int, VarDim, NCORN>();
    auto store2 = data[DataID::ALE_STORE2].device<double, VarDim>();
    auto store3 = data[DataID::ALE_STORE3].device<double, VarDim>();
    auto store4 = data[DataID::ALE_STORE4].device<double, VarDim>();
    auto store5 = data[DataID::ALE_STORE5].device<double, VarDim>();
    auto ndx    = data[DataID::NDX].device<double, VarDim>();
    auto ndy    = data[DataID::NDY].device<double, VarDim>();
    auto fcdv   = data[DataID::ALE_FCDV].device<double, VarDim, NFACE>();

    // Calculate mesh velocity
    kernel::getMeshVelocity(nnd, ale.zeul, store4, store5);

    // Store current position
    utils::kernel::copy<double>(store2, ndx, nnd);
    utils::kernel::copy<double>(store3, ndy, nnd);

    // Construct new position
    geometry::kernel::getVertex(runtime.timestep->dt, store4, store5, ndx, ndy,
            nnd);

    // Construct flux volumes
    geometry::kernel::getFluxVolume(ale.global->zerocut, elnd, store2, store3,
            ndx, ndy, fcdv, nel);
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
