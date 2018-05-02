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
#include "packages/setup/driver/set_thermodynamics.h"

#include <algorithm>

#include "common/constants.h"
#include "common/error.h"
#include "common/sizes.h"
#include "common/data_id.h"
#include "common/data_control.h"
#include "common/view.h"
#include "packages/setup/config.h"
#include "packages/setup/types.h"
#include "packages/setup/kernel/set_thermodynamics.h"



namespace bookleaf {
namespace setup {
namespace driver {

void
setThermodynamics(
        Sizes const &sizes,
        setup::Config const &setup_config,
        DataControl &data,
        Error &err)
{
    auto elreg     = data[DataID::IELREG].chost<int, VarDim>();
    auto elmat     = data[DataID::IELMAT].chost<int, VarDim>();
    auto elvolume  = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto eldensity = data[DataID::ELDENSITY].host<double, VarDim>();
    auto elenergy  = data[DataID::ELENERGY].host<double, VarDim>();
    auto cpmat     = data[DataID::ICPMAT].chost<int, VarDim>();
    auto cpnext    = data[DataID::ICPNEXT].chost<int, VarDim>();
    auto cpdensity = data[DataID::CPDENSITY].host<double, VarDim>();
    auto cpenergy  = data[DataID::CPENERGY].host<double, VarDim>();
    auto mxfcp     = data[DataID::IMXFCP].host<int, VarDim>();
    auto mxncp     = data[DataID::IMXNCP].host<int, VarDim>();
    auto mxel      = data[DataID::IMXEL].host<int, VarDim>();

    auto const &thermo = setup_config.thermo;
    for (auto const &tic : thermo) {
        switch (tic.type) {
        case ThermodynamicsIC::Type::REGION:
            kernel::setThermodynamics(elreg.rows(), tic, elreg, elvolume,
                    eldensity, elenergy);
            break;

        case ThermodynamicsIC::Type::MATERIAL:
            kernel::setThermodynamics(elmat.rows(), tic, elmat, elvolume,
                    eldensity, elenergy);

            if (sizes.ncp > 0) {
                auto cpvolume  = data[DataID::CPVOLUME].chost<double, VarDim>();

                kernel::setThermodynamics(cpmat.rows(), tic, cpmat, cpvolume,
                        cpdensity, cpenergy);
            }
            break;

        default:
            FAIL_WITH_LINE(err, "ERROR: unrecognised IC type");
            return;
        }
    }

    // Check thermodynamics
    bool const any_regions =
        std::any_of(thermo.begin(), thermo.end(),
                [](ThermodynamicsIC const &tic) {
                    return tic.type == ThermodynamicsIC::Type::REGION;
                });

    if (sizes.ncp > 0 && any_regions) {
        kernel::rationaliseThermodynamics(setup_config, sizes.nmx, eldensity,
                elenergy, mxfcp, mxncp, mxel, cpmat, cpnext, cpdensity,
                cpenergy, err);
        if (err.failed()) return;
    }
}

} // namespace driver
} // namespace setup
} // namespace bookleaf
