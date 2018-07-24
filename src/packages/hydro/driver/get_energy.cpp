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

#include "common/runtime.h"
#include "common/sizes.h"
#include "common/timestep.h"
#include "common/data_control.h"
#include "utilities/data/global_configuration.h"
#include "utilities/data/gather.h"
#include "packages/hydro/kernel/get_energy.h"
#include "packages/hydro/config.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
getEnergy(
        hydro::Config const &hydro,
        Runtime const &runtime,
        DataControl &data)
{
    using constants::NCORN;

    double const dt      = runtime.timestep->dts;
    double const zerocut = hydro.global->zerocut;

    int const nel = runtime.sizes->nel;

    auto lag_cnfx = data[DataID::LAG_CNFX].chost<double, VarDim, NCORN>();
    auto lag_cnfy = data[DataID::LAG_CNFY].chost<double, VarDim, NCORN>();
    auto lag_cnu  = data[DataID::LAG_CNU].chost<double, VarDim, NCORN>();
    auto lag_cnv  = data[DataID::LAG_CNV].chost<double, VarDim, NCORN>();
    auto elmass   = data[DataID::ELMASS].chost<double, VarDim>();
    auto elenergy = data[DataID::ELENERGY].host<double, VarDim>();

    // Hydro internal energy update
    hydro::kernel::getEnergy(
            dt,
            zerocut,
            lag_cnfx,
            lag_cnfy,
            lag_cnu,
            lag_cnv,
            elmass,
            elenergy,
            nel);

    int const ncp = runtime.sizes->ncp;
    if (ncp > 0) {
        auto mxel     = data[DataID::IMXEL].chost<int, VarDim>();
        auto mxfcp    = data[DataID::IMXFCP].chost<int, VarDim>();
        auto mxncp    = data[DataID::IMXNCP].chost<int, VarDim>();
        auto lag_cpfx = data[DataID::LAG_CPFX].chost<double, VarDim, NCORN>();
        auto lag_cpfy = data[DataID::LAG_CPFY].chost<double, VarDim, NCORN>();
        auto lag_cpu  = data[DataID::LAG_CPU].host<double, VarDim, NCORN>();
        auto lag_cpv  = data[DataID::LAG_CPV].host<double, VarDim, NCORN>();
        auto cpmass   = data[DataID::CPMASS].chost<double, VarDim>();
        auto cpenergy = data[DataID::CPENERGY].host<double, VarDim>();

        utils::kernel::mxCornerGather(runtime.sizes->nmx, mxel, mxfcp, mxncp,
                lag_cnu, lag_cpu);
        utils::kernel::mxCornerGather(runtime.sizes->nmx, mxel, mxfcp, mxncp,
                lag_cnv, lag_cpv);

        hydro::kernel::getEnergy(
                dt,
                zerocut,
                lag_cpfx,
                lag_cpfy,
                lag_cpu,
                lag_cpv,
                cpmass,
                cpenergy,
                ncp);
    }
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
