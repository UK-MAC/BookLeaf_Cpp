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
#include "infrastructure/init.h"

#include <cassert>

#include "common/config.h"
#include "common/runtime.h"
#include "common/constants.h"
#include "common/error.h"
#include "common/sizes.h"
#include "common/timestep.h"
#include "common/timer_control.h"
#include "packages/init/driver.h"
#include "packages/setup/config.h"
#include "utilities/io/config.h"

#include "utilities/comms/config.h"
#ifdef BOOKLEAF_MPI_SUPPORT
#include "utilities/comms/dt_reduce.h"
#include "utilities/comms/exchange.h"
#endif

#include "utilities/eos/config.h"
#include "utilities/data/global_configuration.h"
#include "common/data_control.h"
#include "packages/time/config.h"
#include "packages/hydro/config.h"
#include "packages/hydro/driver/init.h"
#include "packages/ale/config.h"
#include "packages/setup/driver/set_initial_conditions.h"



namespace bookleaf {
namespace inf {
namespace init {

void
initParallelism(
        comms::Comms &comms)
{
    Error err;
    comms::initComms(comms, err);
    if (err.failed()) {
        assert(false && "unhandled error");
        return;
    }
}



#ifdef BOOKLEAF_MPI_SUPPORT
void
initCommPhases(
        comms::Comm &comm,
        DataControl &data,
        Error &err)
{
    // Create phases
    auto vpid = comms::addCommPhase(comm, "Viscosity", 1, err);
    if (err.failed()) return;

    auto hpid = comms::addCommPhase(comm, "Half Step", 1, err);
    if (err.failed()) return;

    auto aepid = comms::addCommPhase(comm, "Element advection", 2, err);
    if (err.failed()) return;

    auto anpid = comms::addCommPhase(comm, "Nodal advection", 2, err);
    if (err.failed()) return;

    // Attach quantities to phases
    DataID const vqids[] = {
        DataID::TIME_DU,
        DataID::TIME_DV,
        DataID::TIME_DX,
        DataID::TIME_DY,
        DataID::ELDENSITY,
        DataID::ELCS2
    };

    for (DataID data_id : vqids) {
        comms::addDataToCommPhase(comm, vpid, data_id, data, err);
        if (err.failed()) return;
    }

    DataID const hqids[] = {
        DataID::CNMASS,
        DataID::CNWT,
        DataID::LAG_CNFX,
        DataID::LAG_CNFY,
        DataID::ELDENSITY
    };

    for (DataID data_id : hqids) {
        comms::addDataToCommPhase(comm, hpid, data_id, data, err);
        if (err.failed()) return;
    }

    DataID const aeqids[] = {
        DataID::ALE_FCDV,
        DataID::ELDENSITY,
        DataID::ELENERGY,
        DataID::CNMASS,
        DataID::CNWT
    };

    for (DataID data_id : aeqids) {
        comms::addDataToCommPhase(comm, aepid, data_id, data, err);
        if (err.failed()) return;
    }

    DataID const anqids[] = {
        DataID::ALE_FCDV,
        DataID::ALE_FCDM,
        DataID::ALE_CNU,
        DataID::ALE_CNV,
        DataID::ALE_STORE1,
        DataID::ELVOLUME,
        DataID::CNMASS
    };

    for (DataID data_id : anqids) {
        comms::addDataToCommPhase(comm, anpid, data_id, data, err);
        if (err.failed()) return;
    }
}
#endif // ifdef BOOKLEAF_MPI_SUPPORT



void
rationalise(
        Config &config,
        Runtime &runtime,
        Error &err)
{
    // Rationalise runtime
    rationalise(*runtime.sizes, err);
    if (err.failed()) return;

    rationalise(*runtime.timestep, *config.time, err);
    if (err.failed()) return;

    // Rationalise config
    rationalise(*config.eos, runtime.sizes->nmat, err);
    if (err.failed()) return;

    rationalise(*config.io, *runtime.sizes, err);
    if (err.failed()) return;

    rationalise(*config.global, err);
    if (err.failed()) return;

    rationalise(*config.time, err);
    if (err.failed()) return;

    rationalise(*config.hydro, runtime.sizes->nreg, err);
    if (err.failed()) return;

    rationalise(*config.ale, *config.time, err);
    if (err.failed()) return;

    rationalise(*config.setup, err);
    if (err.failed()) return;

    rationaliseMeshDescriptor(*config.setup->mesh_descriptor, *runtime.sizes,
            err);
    if (err.failed()) return;

    rationaliseRegions(
            config.setup->regions,
            config.setup->materials,
            config.setup->shapes,
            err);
    if (err.failed()) return;

    rationaliseMaterials(
            config.setup->materials,
            config.setup->regions,
            config.setup->shapes,
            err);
    if (err.failed()) return;
}



void
init(
        Config &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    // Initialise mpi communicators
    *config.comms->spatial = *config.comms->world;

#ifdef BOOKLEAF_MPI_SUPPORT
    if (config.comms->spatial->nproc > 1) {
        comms::startCommsInit(err);
    }
#endif

    // Initialise mesh
    bookleaf::init::driver::initMesh(config, runtime, timers, data, err);
    if (err.failed()) return;

    // Initialise communication phases and collective types
#ifdef BOOKLEAF_MPI_SUPPORT
    if (config.comms->spatial->nproc > 1) {
        initCommPhases(*config.comms->spatial, data, err);
        if (err.failed()) return;

        comms::initReduceDt(err);
        if (err.failed()) return;

        comms::finishCommsInit(err);
        if (err.failed()) return;
    }
#endif

    // Initialise EOS config
    initEOSConfig(*runtime.sizes, *config.eos, err);
    if (err.failed()) return;

    // Initialise hydro config
    initHydroConfig(*runtime.sizes, *config.hydro, err);
    if (err.failed()) return;

    // Initialise
    bookleaf::init::driver::initElementOrdering(config, *runtime.sizes, data);

    // Initialise initial conditions
    setup::driver::setInitialConditions(*config.setup, *config.global,
            *config.comms->spatial, *runtime.sizes, timers, data, err);
    if (err.failed()) return;

    // Initialise state
    bookleaf::init::driver::initElementState(config, *runtime.sizes, timers,
            data);

    // Initialise hydro
    hydro::driver::init(*config.hydro, *runtime.sizes, data);

    // Sync all data to device
    data.syncAllDevice();
}

} // namespace init
} // namespace inf
} // namespace bookleaf
