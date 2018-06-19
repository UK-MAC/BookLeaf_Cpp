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
#include "packages/init/driver.h"

#include <cassert>

#include "common/config.h"
#include "common/runtime.h"
#include "utilities/comms/config.h"
#include "common/constants.h"
#include "common/error.h"
#include "common/sizes.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "packages/init/kernel.h"
#include "packages/setup/transfer_mesh.h"
#include "utilities/data/gather.h"
#include "utilities/data/sort.h"
#include "packages/ale/config.h"
#include "utilities/eos/get_eos.h"

#ifdef BOOKLEAF_MPI_SUPPORT
#include "packages/setup/partition_mesh.h"
#endif



namespace bookleaf {
namespace init {
namespace driver {
namespace {

/** \brief Initialise element masses. */
void
initElementMasses(
        Sizes const &sizes,
        DataControl &data)
{
    using constants::NCORN;

    auto elmass    = data[DataID::ELMASS].host<double, VarDim>();
    auto cnmass    = data[DataID::CNMASS].host<double, VarDim, NCORN>();
    auto eldensity = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elvolume  = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto cnwt      = data[DataID::CNWT].chost<double, VarDim, NCORN>();

    // Initialise clean cells
    kernel::elMass(sizes.nel, eldensity, elvolume, cnwt, elmass, cnmass);

    // Initialise mixed cells
    if (sizes.ncp > 0) {
        auto mxel       = data[DataID::IMXEL].chost<int, VarDim>();
        auto mxfcp      = data[DataID::IMXFCP].chost<int, VarDim>();
        auto mxncp      = data[DataID::IMXNCP].chost<int, VarDim>();
        auto rcpscratch = data[DataID::RCPSCRATCH11].host<double, VarDim>();
        auto cpdensity  = data[DataID::CPDENSITY].chost<double, VarDim>();
        auto cpvolume   = data[DataID::CPVOLUME].chost<double, VarDim>();
        auto cpmass     = data[DataID::CPMASS].host<double, VarDim>();
        auto frmass     = data[DataID::FRMASS].host<double, VarDim>();

        kernel::mxMass(sizes.ncp, cpdensity, cpvolume, cpmass);

        utils::kernel::mxGather<double>(sizes.nmx, mxel, mxfcp, mxncp, elmass,
                rcpscratch);

        for (int i = 0; i < sizes.ncp; i++) {
            frmass(i) = cpmass(i) / rcpscratch(i);
        }
    }
}

} // namespace

void
initSerialGhosts(
        Sizes &sizes)
{
    sizes.nel1 = sizes.nel;
    sizes.nel2 = sizes.nel;
    sizes.nnd1 = sizes.nnd;
    sizes.nnd2 = sizes.nnd;
}



void
initConnectivity(
        Sizes const &sizes,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    int const nel = sizes.nel2;
    int const nnd = sizes.nnd2;

    // Initialise element-* mappings
    auto elnd = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto elel = data[DataID::IELEL].host<int, VarDim, NFACE>();
    auto elfc = data[DataID::IELFC].host<int, VarDim, NFACE>();

    kernel::getElementConnectivity(elnd, elel, nel);
    kernel::getFaceConnectivity(elel, elfc, nel);
    kernel::correctConnectivity(nel, elel, elfc);

    // Initialise node-element mapping
    auto ndeln = data[DataID::INDELN].host<int, VarDim>();
    auto ndelf = data[DataID::INDELF].host<int, VarDim>();

    kernel::getNodeElementMappingSizes(elnd, ndeln, ndelf, nnd, nel);
    data.setNdEl(sizes);

    auto ndel = data[DataID::INDEL].host<int, VarDim>();

    kernel::getNodeElementMapping(elnd, ndeln, ndelf, ndel, nnd, nel);
}



void
initNodeTypes(
        Sizes &sizes,
        DataControl &data)
{
    using constants::NCORN;

    int const nel = sizes.nel1;
    int const nnd = sizes.nnd1;

    auto elnd   = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto ndtype = data[DataID::INDTYPE].host<int, VarDim>();

    kernel::nodeType(elnd, ndtype, nel, nnd);
}



void
initElementOrdering(
        Config const &config,
        Sizes const &sizes,
        DataControl &data)
{
    auto ellocglob = data[DataID::IELLOCGLOB].chost<int, VarDim>();
    auto elsort1   = data[DataID::IELSORT1].host<int, VarDim>();
    auto elsort2   = data[DataID::IELSORT2].host<int, VarDim>();

    if (config.comms->spatial->nproc > 1) {
        utils::kernel::sortIndices<int, int>(ellocglob, elsort1, sizes.nel1);

        if (config.ale->zexist) {
            utils::kernel::sortIndices<int, int>(ellocglob, elsort2, sizes.nel2);
        }

    } else {
        for (int iel = 0; iel < sizes.nel1; iel++) {
            elsort1(iel) = iel;
        }

        if (config.ale->zexist) {
            for (int iel = 0; iel < sizes.nel2; iel++) {
                elsort2(iel) = iel;
            }
        }
    }

    // Sort node-element mapping by global element number
    auto ndel  = data[DataID::INDEL].host<int, VarDim>();
    auto ndeln = data[DataID::INDELN].chost<int, VarDim>();
    auto ndelf = data[DataID::INDELF].chost<int, VarDim>();

    if (config.comms->spatial->nproc > 1) {
        for (int ind = 0; ind < sizes.nnd2; ind++) {
            int *start = &ndel(ndelf(ind));
            int *end   = start + ndeln(ind);

            std::sort(start, end,
                    [=](int iel, int jel) {
                        return ellocglob(iel) < ellocglob(jel);
                    });
        }

    } else {
        for (int ind = 0; ind < sizes.nnd2; ind++) {
            int *start = &ndel(ndelf(ind));
            int *end   = start + ndeln(ind);

            std::sort(start, end);
        }
    }
}



void
initElementState(
        Config const &config,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data)
{
    // Initialise mass
    initElementMasses(sizes, data);

    // Initialise pressure and sound speed
    data[DataID::IELMAT].syncDevice();
    data[DataID::ELDENSITY].syncDevice();
    data[DataID::ELENERGY].syncDevice();
    data[DataID::ELPRESSURE].syncDevice();
    data[DataID::ELCS2].syncDevice();

    eos::driver::getEOS(*config.eos, sizes, timers, TimerID::GETEOSI, data);

    data[DataID::ELPRESSURE].syncHost();
    data[DataID::ELCS2].syncHost();
}



void
initMesh(
        Config &config,
        Runtime &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    // Transfer data from mesh generation
#ifdef BOOKLEAF_MPI_SUPPORT
    if (config.comms->spatial->nproc > 1) {

        // Partition and transfer mesh
        setup::partitionMesh(config, *config.setup, *runtime.sizes, timers,
                TimerID::MESHPARTITION, data, err);
        if (err.failed()) return;

        // Register data
        data.setQuant(config, *runtime.sizes);
        if (err.failed()) return;

    } else
#endif // BOOKLEAF_MPI_SUPPORT
    {
        // Set null ghost extents
        initSerialGhosts(*runtime.sizes);

        // Register data
        data.setMesh(*runtime.sizes);
        if (err.failed()) return;

        data.setQuant(config, *runtime.sizes);
        if (err.failed()) return;

        // Transfer mesh
        setup::transferMesh(runtime.sizes->nel, *config.setup, data, timers,
                err);
        if (err.failed()) return;
    }

    // Initialise connectivity & node type
    initConnectivity(*runtime.sizes, data);
    initNodeTypes(*runtime.sizes, data);
}

} // namespace driver
} // namespace init
} // namespace bookleaf
