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
#include "common/sizes.h"
#include "common/timestep.h"
#include "packages/hydro/config.h"
#include "common/timer_control.h"
#include "packages/hydro/kernel/get_force.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
getForce(
        hydro::Config const &hydro,
        Runtime const &runtime,
        TimerControl &timers,
        DataControl &data)
{
    using constants::NCORN;

    int const nel = runtime.sizes->nel;
    int const ncp = runtime.sizes->ncp;

    auto elreg      = data[DataID::IELREG].chost<int, VarDim>();
    auto elvolume   = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elcs2      = data[DataID::ELCS2].chost<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].chost<double, VarDim>();
    auto a1         = data[DataID::A1].chost<double, VarDim>();
    auto a3         = data[DataID::A3].chost<double, VarDim>();
    auto b1         = data[DataID::B1].chost<double, VarDim>();
    auto b3         = data[DataID::B3].chost<double, VarDim>();
    auto edviscx    = data[DataID::CNVISCX].chost<double, VarDim, NCORN>();
    auto edviscy    = data[DataID::CNVISCY].chost<double, VarDim, NCORN>();
    auto cnx        = data[DataID::CNX].chost<double, VarDim, NCORN>();
    auto cny        = data[DataID::CNY].chost<double, VarDim, NCORN>();
    auto lag_cnu    = data[DataID::LAG_CNU].chost<double, VarDim, NCORN>();
    auto lag_cnv    = data[DataID::LAG_CNV].chost<double, VarDim, NCORN>();
    auto lag_cnfx   = data[DataID::LAG_CNFX].host<double, VarDim, NCORN>();
    auto lag_cnfy   = data[DataID::LAG_CNFY].host<double, VarDim, NCORN>();

    auto cppressure = data[DataID::CPPRESSURE].chost<double, VarDim>();
    auto cpa1       = data[DataID::CPA1].chost<double, VarDim>();
    auto cpa3       = data[DataID::CPA3].chost<double, VarDim>();
    auto cpb1       = data[DataID::CPB1].chost<double, VarDim>();
    auto cpb3       = data[DataID::CPB3].chost<double, VarDim>();
    auto cpviscx    = data[DataID::CPVISCX].chost<double, VarDim, NCORN>();
    auto cpviscy    = data[DataID::CPVISCY].chost<double, VarDim, NCORN>();
    auto lag_cpfx   = data[DataID::LAG_CPFX].host<double, VarDim, NCORN>();
    auto lag_cpfy   = data[DataID::LAG_CPFY].host<double, VarDim, NCORN>();

    // Pressure force
    kernel::getForcePressure(elpressure, a1, a3, b1, b3, lag_cnfx, lag_cnfy, nel);
    if (ncp > 0) {
        kernel::getForcePressure(cppressure, cpa1, cpa3, cpb1, cpb3, lag_cpfx,
                lag_cpfy, ncp);
    }

    // Artificial viscosity force
    kernel::getForceViscosity(edviscx, edviscy, lag_cnfx, lag_cnfy, nel);
    if (ncp > 0) {
        kernel::getForceViscosity(cpviscx, cpviscy, lag_cpfx, lag_cpfy, ncp);
    }

    // Subzonal pressure force
    if (hydro.zsp) {
        auto spmass = data[DataID::SPMASS].chost<double, VarDim, NCORN>();

        ScopedTimer st(timers, TimerID::GETSP);
        kernel::getForceSubzonalPressure(
                hydro.pmeritreg.data(),
                elreg,
                eldensity,
                elcs2,
                cnx,
                cny,
                spmass,
                lag_cnfx,
                lag_cnfy,
                nel);
    }

    // Anti-hourglass force
    if (runtime.timestep->zcorrector && hydro.zhg) {
        ScopedTimer st(timers, TimerID::GETHG);
        kernel::getForceHourglass(
                runtime.timestep->dt,
                hydro.kappareg.data(),
                elreg,
                eldensity,
                elvolume,
                lag_cnu,
                lag_cnv,
                lag_cnfx,
                lag_cnfy,
                nel);
    }
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
