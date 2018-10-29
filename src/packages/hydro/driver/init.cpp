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
#include "packages/hydro/driver/init.h"

#include <cassert>

#include "packages/hydro/config.h"
#include "packages/hydro/kernel/init.h"
#include "common/sizes.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
init(hydro::Config &hydro, Sizes const &sizes, DataControl &data)
{
    using constants::NCORN;

    auto elvisc  = data[DataID::ELVISC].host<double, VarDim>();
    auto edviscx = data[DataID::CNVISCX].host<double, VarDim, NCORN>();
    auto edviscy = data[DataID::CNVISCY].host<double, VarDim, NCORN>();

    // Initialise artificial viscosity
    kernel::initViscosity(sizes.nel2, elvisc, edviscx, edviscy);

    if (sizes.ncp > 0) {
        auto cpvisc  = data[DataID::CPVISC].host<double, VarDim>();
        auto cpviscx = data[DataID::CPVISCX].host<double, VarDim, NCORN>();
        auto cpviscy = data[DataID::CPVISCY].host<double, VarDim, NCORN>();

        kernel::initViscosity(sizes.ncp, cpvisc, cpviscx, cpviscy);
    }

    // Initialise subzonal pressure mass
    if (hydro.zsp) {
        auto eldensity = data[DataID::ELDENSITY].chost<double, VarDim>();
        auto cnx       = data[DataID::CNX].chost<double, VarDim, NCORN>();
        auto cny       = data[DataID::CNY].chost<double, VarDim, NCORN>();
        auto spmass    = data[DataID::SPMASS].host<double, VarDim, NCORN>();

        kernel::initSubzonalPressureMass(sizes.nel, eldensity, cnx, cny,
                spmass);
    }
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
