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

    auto elreg      = data[DataID::IELREG].cdevice<int, VarDim>();
    auto elvolume   = data[DataID::ELVOLUME].cdevice<double, VarDim>();
    auto eldensity  = data[DataID::ELDENSITY].cdevice<double, VarDim>();
    auto elcs2      = data[DataID::ELCS2].cdevice<double, VarDim>();
    auto elpressure = data[DataID::ELPRESSURE].cdevice<double, VarDim>();
    auto a1         = data[DataID::A1].cdevice<double, VarDim>();
    auto a3         = data[DataID::A3].cdevice<double, VarDim>();
    auto b1         = data[DataID::B1].cdevice<double, VarDim>();
    auto b3         = data[DataID::B3].cdevice<double, VarDim>();
    auto edviscx    = data[DataID::CNVISCX].cdevice<double, VarDim, NCORN>();
    auto edviscy    = data[DataID::CNVISCY].cdevice<double, VarDim, NCORN>();
    auto cnx        = data[DataID::CNX].cdevice<double, VarDim, NCORN>();
    auto cny        = data[DataID::CNY].cdevice<double, VarDim, NCORN>();
    auto lag_cnu    = data[DataID::LAG_CNU].cdevice<double, VarDim, NCORN>();
    auto lag_cnv    = data[DataID::LAG_CNV].cdevice<double, VarDim, NCORN>();
    auto lag_cnfx   = data[DataID::LAG_CNFX].device<double, VarDim, NCORN>();
    auto lag_cnfy   = data[DataID::LAG_CNFY].device<double, VarDim, NCORN>();

    auto cppressure = data[DataID::CPPRESSURE].cdevice<double, VarDim>();
    auto cpa1       = data[DataID::CPA1].cdevice<double, VarDim>();
    auto cpa3       = data[DataID::CPA3].cdevice<double, VarDim>();
    auto cpb1       = data[DataID::CPB1].cdevice<double, VarDim>();
    auto cpb3       = data[DataID::CPB3].cdevice<double, VarDim>();
    auto cpviscx    = data[DataID::CPVISCX].cdevice<double, VarDim, NCORN>();
    auto cpviscy    = data[DataID::CPVISCY].cdevice<double, VarDim, NCORN>();
    auto lag_cpfx   = data[DataID::LAG_CPFX].device<double, VarDim, NCORN>();
    auto lag_cpfy   = data[DataID::LAG_CPFY].device<double, VarDim, NCORN>();

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
        auto spmass = data[DataID::SPMASS].cdevice<double, VarDim, NCORN>();

        ConstDeviceView<double, VarDim> pmeritreg(
                hydro.device_pmeritreg, runtime.sizes->nreg);

        ScopedTimer st(timers, TimerID::GETSP);
        kernel::getForceSubzonalPressure(
                pmeritreg,
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

        ConstDeviceView<double, VarDim> kappareg(
                hydro.device_kappareg, runtime.sizes->nreg);

        kernel::getForceHourglass(
                runtime.timestep->dt,
                kappareg,
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
