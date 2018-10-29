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
#include "packages/setup/driver/set_initial_conditions.h"

#include <stdio.h>
#include <algorithm>
#include <cmath>
#include <cassert>

#include "common/constants.h"
#include "common/timer_control.h"
#include "utilities/comms/config.h"
#include "common/data_control.h"
#include "utilities/geometry/geometry.h"
#include "utilities/misc/boundary_conditions.h"

#include "utilities/data/global_configuration.h"
#include "utilities/mix/driver/list.h"
#include "utilities/data/gather.h"

#include "common/error.h"
#include "common/sizes.h"

#include "packages/setup/config.h"
#include "packages/setup/indicators.h"
#include "packages/setup/types.h"
#include "packages/setup/driver/set_thermodynamics.h"
#include "packages/setup/driver/set_kinematics.h"

#include "packages/setup/kernel/set_flags.h"
#include "packages/setup/driver/set_shape_flags.h"



namespace bookleaf {
namespace setup {
namespace driver {
namespace {

using constants::NCORN;

void
setRegionFlags(
        setup::Config &setup_config,
        comms::Comm const &comm,
        Sizes &sizes,
        DataControl &data,
        Error &err)
{
    auto flag = data[DataID::IELREG].host<int, VarDim>();

    if (!setup_config.rmesh) {
        for (int iel = 0; iel < sizes.nel; iel++) {
            flag(iel) = -1;
        }
    }

    for (auto const &region : setup_config.regions) {

        switch (region.type) {
        case Indicator::Type::MESH:
            // Do nothing, handled elsewhere. XXX Note return, ignore all
            // remaining indicators (they're assumed to also be of type MESH).
            return;

        case Indicator::Type::BACKGROUND:
            kernel::setFlag(
                    flag.size(),
                    region.index,
                    flag);
            break;

        case Indicator::Type::CELL:
            {
                // If we are running with MPI, we need to check the global cell
                // (element) numbering to set the right flags. If not, we just
                // use the local numbering. IELMAT is just a dummy here, it is
                // never used.
                DataID const ipid = comm.nproc > 1 ?
                                        DataID::IELLOCGLOB : DataID::IELMAT;
                auto ip = data[ipid].chost<int, VarDim>();

                kernel::flag::setCellFlags(
                        flag.size(),
                        region.index,
                        region.value,
                        comm.nproc,
                        ip,
                        flag,
                        err);
                if (err.failed()) return;
            }
            break;

        case Indicator::Type::SHAPE:
            setShapeFlags(setup_config.shapes, region.index, region.value,
                    DataID::IELREG, sizes, data, err, applyShapeRegion);
            if (err.failed()) return;
            break;

        case Indicator::Type::REGION:
            err.fail("ERROR: region tagged incorrectly");
            return;

        default:
            err.fail("ERROR: unrecognised indicator type");
            return;
        }
    }

    // Check all elements have a region set
    for (int iel = 0; iel < sizes.nel; iel++) {
        if (flag(iel) == -1) {
            FAIL_WITH_LINE(err, "ERROR: region not populated");
            return;
        }
    }
}



void
setMaterialFlags(
        setup::Config &setup_config,
        comms::Comm const &comm,
        Sizes &sizes,
        DataControl &data,
        Error &err)
{
    int const nmat = sizes.nmat;

    auto test = data[DataID::IELREG].chost<int, VarDim>();
    auto flag = data[DataID::IELMAT].host<int, VarDim>();

    // We use negative numbers to represent multi-material elements, so
    // rather than using -1 to represent an unpopulated entry, use one
    // greater than the largest material index
    if (!setup_config.mmesh) {
        for (int iel = 0; iel < sizes.nel; iel++) {
            flag(iel) = nmat;
        }
    }

    for (auto const &material : setup_config.materials) {

        switch (material.type) {
        case Indicator::Type::MESH:
            // Do nothing, handled elsewhere. XXX Note return, ignore all
            // remaining indicators (they're assumed to also be of type MESH).
            return;

        case Indicator::Type::BACKGROUND:
            kernel::setFlag(
                    flag.size(),
                    material.index,
                    flag);
            break;

        case Indicator::Type::CELL:
            {
                DataID const ipid = comm.nproc > 1 ?
                                        DataID::IELLOCGLOB : DataID::IELREG;
                auto ip = data[ipid].chost<int, VarDim>();

                kernel::flag::setCellFlags(
                        flag.size(),
                        material.index,
                        material.value,
                        comm.nproc,
                        ip,
                        flag,
                        err);
                if (err.failed()) return;
            }
            break;

        case Indicator::Type::SHAPE:
            setShapeFlags(setup_config.shapes, material.index, material.value, 
                    DataID::IELMAT, sizes, data, err, applyShapeMaterial);
            if (err.failed()) return;
            break;

        case Indicator::Type::REGION:
            kernel::setFlagIf(
                    flag.size(),
                    material.index,
                    material.value,
                    test,
                    flag);
            break;

        default:
            err.fail("ERROR: unrecognised indicator type");
            return;
        }
    }

    // Check all elements have a material set
    for (int iel = 0; iel < sizes.nel; iel++) {
        if (flag(iel) == nmat) {
            FAIL_WITH_LINE(err, "ERROR: material not populated");
            return;
        }
    }

    if (sizes.ncp > 0) {
        auto cpmat = data[DataID::ICPMAT].chost<int, VarDim>();
        for (int icp = 0; icp < sizes.ncp; icp++) {
            if (cpmat(icp) == nmat) {
                FAIL_WITH_LINE(err, "ERROR: material (multi) not populated");
                return;
            }
        }
    }
}



void
setState(
        geometry::Config const &geom,
        setup::Config &setup_config,
        GlobalConfiguration &global,
        Sizes &sizes,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    auto eldensity = data[DataID::ELDENSITY].host<double, VarDim>();
    auto elenergy  = data[DataID::ELENERGY].host<double, VarDim>();

    // Get geometry
    data[DataID::IELND].syncDevice();
    data[DataID::NDX].syncDevice();
    data[DataID::NDY].syncDevice();

    geometry::driver::getGeometry(geom, sizes, timers, TimerID::GETGEOMETRYI,
            data, err);
    if (err.failed()) return;

    data[DataID::CNX].syncHost();
    data[DataID::CNY].syncHost();
    data[DataID::CNWT].syncHost();
    data[DataID::A1].syncHost();
    data[DataID::A2].syncHost();
    data[DataID::A3].syncHost();
    data[DataID::B1].syncHost();
    data[DataID::B2].syncHost();
    data[DataID::B3].syncHost();
    data[DataID::ELVOLUME].syncHost();

    // Set thermodynamics
    if (!setup_config.thermo.empty()) {
        setThermodynamics(sizes, setup_config, data, err);
        if (err.failed()) return;
    }

    // Set kinematics
    auto ndu = data[DataID::NDU].host<double, VarDim>();
    auto ndv = data[DataID::NDV].host<double, VarDim>();

    for (int ind = 0; ind < sizes.nnd; ind++) {
        ndu(ind) = 0.;
        ndv(ind) = 0.;
    }

    if (!setup_config.kinematics.empty()) {
        setKinematics(sizes, global, setup_config.kinematics, data);
    }

    // Enforce boundary conditions
    data[DataID::INDTYPE].syncDevice();
    data[DataID::NDU].syncDevice();
    data[DataID::NDV].syncDevice();

    utils::driver::setBoundaryConditions(global, sizes, DataID::NDU,
            DataID::NDV, data);

    data[DataID::NDU].syncHost();
    data[DataID::NDV].syncHost();

    // Check mesh is fully populated
    for (int iel = 0; iel < sizes.nel; iel++) {
        if (eldensity(iel) <= 0.) {
            FAIL_WITH_LINE(err, "ERROR: eldensity not populated");
            return;
        }
    }

    for (int iel = 0; iel < sizes.nel; iel++) {
        if (elenergy(iel) < 0.) {
            FAIL_WITH_LINE(err, "ERROR: elenergy not populated");
            return;
        }
    }

    if (sizes.ncp > 0) {
        auto cpdensity = data[DataID::CPDENSITY].chost<double, VarDim>(sizes.ncp);
        auto cpenergy  = data[DataID::CPENERGY].chost<double, VarDim>(sizes.ncp);

        for (int icp = 0; icp < sizes.ncp; icp++) {
            if (cpdensity(icp) <= 0.) {
                FAIL_WITH_LINE(err, "ERROR: cpdensity not populated");
                return;
            }
        }

        for (int icp = 0; icp < sizes.ncp; icp++) {
            if (cpenergy(icp) < 0.) {
                FAIL_WITH_LINE(err, "ERROR: cpenergy not populated");
                return;
            }
        }
    }

    // Tidy up
    setup_config.thermo.clear();
    setup_config.kinematics.clear();
    setup_config.regions.clear();
    setup_config.materials.clear();
    setup_config.shapes.clear();
}

} // namespace

void
setInitialConditions(
        geometry::Config const &geom,
        setup::Config &setup_config,
        GlobalConfiguration &global,
        comms::Comm &comm,
        Sizes &sizes,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    ScopedTimer st(timers, TimerID::SETUPIC);

    // Determine which elements belong to which regions
    setRegionFlags(setup_config, comm, sizes, data, err);
    if (err.failed()) return;

    // Determine which elements contain which materials
    setMaterialFlags(setup_config, comm, sizes, data, err);
    if (err.failed()) return;

    // Set initial conditions (density, energy, velocity etc.) based on region
    // and materials per element.
    setState(geom, setup_config, global, sizes, timers, data, err);
    if (err.failed()) return;
}

} // namespace driver
} // namespace setup
} // namespace bookleaf
