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
#include "packages/setup/driver/set_kinematics.h"

#include <algorithm>

#include "common/constants.h"
#include "common/sizes.h"
#include "common/data_id.h"
#include "common/data_control.h"
#include "common/view.h"
#include "utilities/data/global_configuration.h"
#include "packages/setup/types.h"
#include "packages/setup/kernel/set_kinematics.h"



namespace bookleaf {
namespace setup {
namespace driver {

void
setKinematics(
        Sizes const &sizes,
        GlobalConfiguration const &global,
        std::vector<KinematicsIC> const &kinematic,
        DataControl &data)
{
    using constants::NCORN;

    bool const any_regions = std::any_of(kinematic.begin(), kinematic.end(),
            [](KinematicsIC const &kic) -> bool {
                return kic.type == KinematicsIC::Type::REGION;
            });

    auto elreg      = data[DataID::IELREG].chost<int, VarDim>();
    auto elnd       = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto cnwt       = data[DataID::CNWT].chost<double, VarDim, NCORN>();
    auto cnx        = data[DataID::SETUP_CNX].chost<double, VarDim, NCORN>();
    auto cny        = data[DataID::SETUP_CNY].chost<double, VarDim, NCORN>();
    auto ndx        = data[DataID::NDX].chost<double, VarDim>();
    auto ndy        = data[DataID::NDY].chost<double, VarDim>();
    auto ndu        = data[DataID::NDU].host<double, VarDim>();
    auto ndv        = data[DataID::NDV].host<double, VarDim>();
    auto rscratch11 = data[DataID::RSCRATCH11].host<double, VarDim>();

    if (any_regions) {
        for (int i = 0; i < sizes.nnd2; i++) {
            rscratch11(i) = 0.;
        }
    }

    for (auto const &kic: kinematic) {
        switch (kic.type) {
        case KinematicsIC::Type::BACKGROUND:
            kernel::setBackgroundKinematics(sizes.nnd, kic, ndx, ndy, ndu, ndv);
            break;

        case KinematicsIC::Type::REGION:
            kernel::setRegionKinematics(sizes.nel1, sizes.nnd1, kic, elreg, elnd, cnwt, cnx, cny,
                    ndu, ndv, rscratch11);
            break;

        default:
            // Do nothing
            break;
        }
    }

    if (any_regions) {
        kernel::rationaliseKinematics(sizes.nnd, global.zerocut, rscratch11, ndu, ndv);
    }
}

} // namespace driver
} // namespace setup
} // namespace bookleaf
