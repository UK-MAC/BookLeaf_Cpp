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

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/sizes.h"
#include "common/runtime.h"
#include "common/timestep.h"
#include "packages/hydro/config.h"
#include "packages/hydro/kernel/get_acceleration.h"
#include "utilities/data/global_configuration.h"
#include "common/data_control.h"
#include "utilities/misc/boundary_conditions.h"
#include "utilities/data/gather.h"
#include "utilities/misc/average.h"
#include "common/timer_control.h"

#include "utilities/comms/config.h"
#ifdef BOOKLEAF_MPI_SUPPORT
#include "utilities/comms/exchange.h"
#endif



namespace bookleaf {
namespace hydro {
namespace driver {

void
getAcceleration(
        hydro::Config const &hydro,
        Runtime const &runtime,
#ifdef BOOKLEAF_MPI_SUPPORT
        TimerControl &timers,
#else
        TimerControl &timers __attribute__((unused)),
#endif
        DataControl &data)
{
    if (runtime.sizes->ncp > 0) {
        // Average forces
        utils::driver::average(
                *runtime.sizes,
                DataID::FRVOLUME,
                DataID::LAG_CPFX,
                DataID::LAG_CPFY,
                DataID::LAG_CNFX,
                DataID::LAG_CNFY,
                data);
    }

#ifdef BOOKLEAF_MPI_SUPPORT
    if (hydro.comm->nproc > 1) {
        Error err;
        comms::exchange(*hydro.comm, 1, TimerID::COMMT, timers, data, err);
        if (err.failed()) {
            assert(false && "unhandled error");
        }
    }
#endif

    using constants::NCORN;

    int const nnd1 = runtime.sizes->nnd1;
    int const nnd  = runtime.sizes->nnd;

    auto ndeln      = data[DataID::INDELN].chost<int, VarDim>();
    auto ndelf      = data[DataID::INDELF].chost<int, VarDim>();
    auto ndel       = data[DataID::INDEL].chost<int, VarDim>();
    auto elnd       = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto cnwt       = data[DataID::CNWT].chost<double, VarDim, NCORN>();
    auto cnmass     = data[DataID::CNMASS].chost<double, VarDim, NCORN>();
    auto lag_cnfx   = data[DataID::LAG_CNFX].chost<double, VarDim, NCORN>();
    auto lag_cnfy   = data[DataID::LAG_CNFY].chost<double, VarDim, NCORN>();
    auto lag_ndarea = data[DataID::LAG_NDAREA].host<double, VarDim>();
    auto lag_ndmass = data[DataID::LAG_NDMASS].host<double, VarDim>();
    auto lag_ndubar = data[DataID::LAG_NDUBAR].host<double, VarDim>();
    auto lag_ndvbar = data[DataID::LAG_NDVBAR].host<double, VarDim>();
    auto ndu        = data[DataID::NDU].host<double, VarDim>();
    auto ndv        = data[DataID::NDV].host<double, VarDim>();

    kernel::scatterAcceleration(hydro.global->zerocut, ndeln, ndelf,
            ndel, elnd, eldensity, cnwt, cnmass, lag_cnfx, lag_cnfy, lag_ndarea,
            lag_ndmass, lag_ndubar, lag_ndvbar, nnd1);

    kernel::getAcceleration(hydro.global->dencut, hydro.global->zerocut,
            lag_ndarea, lag_ndmass, lag_ndubar, lag_ndvbar, runtime.sizes->nnd);

    // Apply boundary conditions
    utils::driver::setBoundaryConditions(*hydro.global, *runtime.sizes,
            DataID::LAG_NDUBAR, DataID::LAG_NDVBAR, data);

    // XXX Missing code here that can't be merged

    kernel::applyAcceleration(runtime.timestep->dt, lag_ndubar, lag_ndvbar, ndu,
            ndv, nnd);

    // Gather
    utils::driver::cornerGather(*runtime.sizes, DataID::LAG_NDUBAR,
            DataID::LAG_CNU, data);
    utils::driver::cornerGather(*runtime.sizes, DataID::LAG_NDVBAR,
            DataID::LAG_CNV, data);

    // XXX Missing code here that can't be merged
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
