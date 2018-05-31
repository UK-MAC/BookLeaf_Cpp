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
#include "packages/ale/driver/advect.h"

#include <cassert>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/runtime.h"
#include "common/timestep.h"
#include "common/sizes.h"
#include "packages/ale/config.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "utilities/data/global_configuration.h"
#include "utilities/data/gather.h"
#include "utilities/data/copy.h"
#include "packages/ale/kernel/advectors.h"
#include "packages/ale/kernel/advect.h"

#include "utilities/comms/config.h"
#ifdef BOOKLEAF_MPI_SUPPORT
#include "utilities/comms/exchange.h"
#endif



namespace bookleaf {
namespace ale {
namespace driver {
namespace {

void
advectBasisEl(
        int id1,
        int id2,
        ale::Config const &ale,
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, timerid);

    auto elel      = data[DataID::IELEL].chost<int, VarDim, NFACE>();
    auto elfc      = data[DataID::IELFC].chost<int, VarDim, NFACE>();
    auto cnwt      = data[DataID::CNWT].chost<double, VarDim, NCORN>();
    auto fcdv      = data[DataID::ALE_FCDV].chost<double, VarDim, NFACE>();
    auto fcdm      = data[DataID::ALE_FCDM].host<double, VarDim, NFACE>();
    auto elvolume  = data[DataID::ELVOLUME].host<double, VarDim>();
    auto elmass    = data[DataID::ELMASS].host<double, VarDim>();
    auto eldensity = data[DataID::ELDENSITY].host<double, VarDim>();
    auto rwork1    = data[DataID::ALE_RWORK1].host<double, VarDim>();
    auto rwork2    = data[DataID::ALE_RWORK2].host<double, VarDim>();
    auto store1    = data[DataID::ALE_STORE1].host<double, VarDim>();
    auto store2    = data[DataID::ALE_STORE2].host<double, VarDim>();
    auto store3    = data[DataID::ALE_STORE3].host<double, VarDim>();
    auto store5    = data[DataID::ALE_STORE5].host<double, VarDim>();
    auto store6    = data[DataID::ALE_STORE6].host<double, VarDim>();

    // Calculate total volume flux
    kernel::sumFlux(id1, id2, sizes.nel, sizes.nel1, elel, elfc, fcdv, rwork1);

    // Construct mass flux
    kernel::fluxElVl(id1, id2, sizes.nel1, sizes.nel2, elel, elfc, cnwt, fcdv,
            eldensity, fcdm);

    // Calculate total mass flux
    kernel::sumFlux(id1, id2, sizes.nel, sizes.nel1, elel, elfc, fcdm, rwork2);

    // Update
    kernel::updateBasisEl(ale.global->zerocut, ale.global->dencut, rwork1,
            rwork2, store5, store6, store1, store2, store3, elvolume, elmass,
            eldensity, sizes.nel);
}



void
advectVarEl(
        int id1,
        int id2,
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, timerid);

    auto elel     = data[DataID::IELEL].chost<int, VarDim, NFACE>();
    auto elfc     = data[DataID::IELFC].chost<int, VarDim, NFACE>();
    auto elmass   = data[DataID::ELMASS].chost<double, VarDim>();
    auto cnmass   = data[DataID::CNMASS].chost<double, VarDim, NCORN>();
    auto elenergy = data[DataID::ELENERGY].host<double, VarDim>();
    auto fcdm     = data[DataID::ALE_FCDM].chost<double, VarDim, NFACE>();
    auto flux     = data[DataID::ALE_FLUX].host<double, VarDim, NFACE>();
    auto store2   = data[DataID::ALE_STORE2].chost<double, VarDim>();
    auto store6   = data[DataID::ALE_STORE6].chost<double, VarDim>();
    auto rwork1   = data[DataID::ALE_RWORK1].host<double, VarDim>();

    // Internal energy (mass weighted)
    kernel::fluxElVl(id1, id2, sizes.nel1, sizes.nel2, elel, elfc, cnmass, fcdm,
            elenergy, flux);

    kernel::updateEl(id1, id2, sizes.nel, sizes.nel1, elel, elfc, store2,
            elmass, store6, flux, rwork1, elenergy);
}



void
advectBasisNd(
        int id1,
        int id2,
        ale::Config const &ale,
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, timerid);

    auto elnd     = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto elel     = data[DataID::IELEL].chost<int, VarDim, NFACE>();
    auto elfc     = data[DataID::IELFC].chost<int, VarDim, NFACE>();
    auto elsort   = data[DataID::IELSORT2].chost<int, VarDim>();
    auto elvolume = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto cnmass   = data[DataID::CNMASS].host<double, VarDim, NCORN>();
    auto fcdv     = data[DataID::ALE_FCDV].chost<double, VarDim, NFACE>();
    auto fcdm     = data[DataID::ALE_FCDM].chost<double, VarDim, NFACE>();
    auto flux     = data[DataID::ALE_FLUX].host<double, VarDim, NFACE>();
    auto store1   = data[DataID::ALE_STORE1].host<double, VarDim>();
    auto store2   = data[DataID::ALE_STORE2].host<double, VarDim>();
    auto store3   = data[DataID::ALE_STORE3].host<double, VarDim>();
    auto store4   = data[DataID::ALE_STORE4].host<double, VarDim>();
    auto store5   = data[DataID::ALE_STORE5].host<double, VarDim>();
    auto store6   = data[DataID::ALE_STORE6].host<double, VarDim>();
    auto rwork1   = data[DataID::ALE_RWORK1].host<double, VarDim, NCORN>();
    auto rwork2   = data[DataID::ALE_RWORK2].host<double, VarDim, NCORN>();
    auto rwork3   = data[DataID::ALE_RWORK3].host<double, VarDim, NCORN>();

    // Initialise
    kernel::initBasisNd(store3, store4, store2, sizes.nnd2);

    // Construct pre/post nodal volumes and pre nodal/corner mass
    kernel::calcBasisNd(elnd, elsort, store1, elvolume, cnmass, store3,
            store4, store2, rwork3, sizes.nel2);

    // Construct volume and mass flux
    kernel::fluxBasisNd(id1, id2, elel, elfc, elsort, fcdv, fcdm, rwork1,
            rwork2, flux, sizes.nel2);

    // Construct post nodal/corner mass
    utils::kernel::copy<double>(store1, store2, sizes.nnd2);
    kernel::massBasisNd(elnd, elsort, flux, cnmass, store1, sizes.nel2);

    // Construct cut-offs
    kernel::cutBasisNd(ale.global->zerocut, ale.global->dencut, store3, store5,
            store6, sizes.nnd);
}



void
advectVarNd(
        Sizes const &sizes,
        TimerControl &timers,
        TimerID timerid,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, timerid);

    auto elnd     = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto elel     = data[DataID::IELEL].chost<int, VarDim, NFACE>();
    auto elfc     = data[DataID::IELFC].chost<int, VarDim, NFACE>();
    auto cnu      = data[DataID::ALE_CNU].chost<double, VarDim, NCORN>();
    auto cnv      = data[DataID::ALE_CNV].chost<double, VarDim, NCORN>();
    auto ndu      = data[DataID::NDU].host<double, VarDim>();
    auto ndv      = data[DataID::NDV].host<double, VarDim>();
    auto fcdm     = data[DataID::ALE_FCDM].host<double, VarDim>();
    auto flux     = data[DataID::ALE_FLUX].host<double, VarDim, NFACE>();
    auto ndstatus = data[DataID::ALE_INDSTATUS].chost<int, VarDim>();
    auto ndtype   = data[DataID::INDTYPE].chost<int, VarDim>();
    auto active   = data[DataID::ALE_ZACTIVE].host<unsigned char, VarDim>();
    auto store1   = data[DataID::ALE_STORE1].chost<double, VarDim>();
    auto store2   = data[DataID::ALE_STORE2].chost<double, VarDim>();
    auto store6   = data[DataID::ALE_STORE6].chost<double, VarDim>();
    auto rwork2   = data[DataID::ALE_RWORK2].chost<double, VarDim, NCORN>();
    auto rwork3   = data[DataID::ALE_RWORK3].chost<double, VarDim, NCORN>();

    // Momentum (mass weighted)
    kernel::activeNd(-1, ndstatus, ndtype, active, sizes.nnd);

    kernel::fluxNdVl(sizes.nel1, sizes.nel2, elel, elfc, rwork3, rwork2, cnu,
            flux);

    kernel::updateNd(sizes.nnd, sizes.nel1, sizes.nnd1, elnd, store2, store1,
            store6, active, flux, fcdm, ndu);

    kernel::activeNd(-2, ndstatus, ndtype, active, sizes.nnd);

    kernel::fluxNdVl(sizes.nel1, sizes.nel2, elel, elfc, rwork3, rwork2, cnv,
            flux);

    kernel::updateNd(sizes.nnd, sizes.nel1, sizes.nnd1, elnd, store2, store1,
            store6, active, flux, fcdm, ndv);
}



void
advectEl(
        int id1,
        int id2,
        ale::Config const &ale,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::ALEADVECTEL);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (ale.comm->nproc > 1) {
        Error err;
        comms::exchange(*ale.comm, 2, TimerID::COMMA, timers, data, err);
        if (err.failed()) {
            assert(false && "unhandled error");
        }
    }
#endif

    // Advect element basis variables
    advectBasisEl(id1, id2, ale, sizes, timers, TimerID::ALEADVECTBASISEL, data);

    // Advect element independent variables
    advectVarEl(id1, id2, sizes, timers, TimerID::ALEADVECTVAREL, data);
}



void
advectNd(
        int id1,
        int id2,
        ale::Config const &ale,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data)
{
    ScopedTimer st(timers, TimerID::ALEADVECTND);

    utils::driver::cornerGather(sizes, DataID::NDU, DataID::ALE_CNU, data);
    utils::driver::cornerGather(sizes, DataID::NDV, DataID::ALE_CNV, data);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (ale.comm->nproc > 1) {
        Error err;
        comms::exchange(*ale.comm, 3, TimerID::COMMA, timers, data, err);
        if (err.failed()) {
            assert(false && "unhandled error");
        }
    }
#endif

    // Advect nodal basis variables
    advectBasisNd(id1, id2, ale, sizes, timers, TimerID::ALEADVECTBASISND, data);

    // Advect nodal independent variables
    advectVarNd(sizes, timers, TimerID::ALEADVECTVARND, data);
}

} // namespace

void
advect(
        ale::Config const &ale,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data,
        Error &err)
{
    ScopedTimer st(timers, TimerID::ALEADVECT);

    // Advect
    switch(ale.adv_type) {
    case 1: // Isotropic advection
        // Advect element variables
        advectEl(1, 2, ale, *runtime.sizes, timers, data);

        // Advect nodal variables
        advectNd(1, 2, ale, *runtime.sizes, timers, data);
        break;

    case 2: // Split advection
        {
            int ii = (runtime.timestep->nstep + 1) % 2;
            int i1 = 1 + ii;
            int i2 = 2 - ii;
            int i3 = i2 - i1;
            for (int ii = i1; ii <= i2; ii += i3) {
                // Advect element variables
                advectEl(ii, ii, ale, *runtime.sizes, timers, data);

                // Advect nodal variables
                advectNd(ii, ii, ale, *runtime.sizes, timers, data);
            }
        }
        break;

    default:
        err.fail("ERROR: unrecognised adv_type");
    }
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
