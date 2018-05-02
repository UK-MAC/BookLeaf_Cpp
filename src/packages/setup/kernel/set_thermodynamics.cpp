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
#include "packages/setup/kernel/set_thermodynamics.h"

#include <memory>

#include "common/error.h"
#include "packages/setup/config.h"
#include "packages/setup/types.h"



namespace bookleaf {
namespace setup {
namespace kernel {

void
setThermodynamics(
        int nsize,
        ThermodynamicsIC const &tic,
        ConstView<int, VarDim>    flag,
        ConstView<double, VarDim> volume,
        View<double, VarDim>      density,
        View<double, VarDim>      energy)
{
    for (int i = 0; i < nsize; i++) {
        if (flag(i) == tic.value) {
            density(i) = tic.density;
            energy(i) = tic.energy;
        }
    }

    if (tic.energy_scale == ThermodynamicsIC::EnergyScale::VOLUME) {
        for (int i = 0; i < nsize; i++) {
            if (flag(i) == tic.value) {
                energy(i) /= volume(i);
            }
        }

    } else if (tic.energy_scale == ThermodynamicsIC::EnergyScale::MASS) {
        for (int i = 0; i < nsize; i++) {
            if (flag(i) == tic.value) {
                energy(i) /= density(i) * volume(i);
            }
        }
    }
}



void
rationaliseThermodynamics(
        setup::Config const &setup_config,
        int nmx,
        ConstView<double, VarDim> eldensity,
        ConstView<double, VarDim> elenergy,
        ConstView<int, VarDim>    mxfcp,
        ConstView<int, VarDim>    mxncp,
        ConstView<int, VarDim>    mxel,
        ConstView<int, VarDim>    cpmat,
        ConstView<int, VarDim>    cpnext,
        View<double, VarDim>      cpdensity,
        View<double, VarDim>      cpenergy,
        Error &err)
{
    int const nic = setup_config.thermo.size();

    std::unique_ptr<int[]> _ilocal(new int[nic]);
    View<int, VarDim> local(_ilocal.get(), nic);

    for (int i = 0; i < nic; i++) {
        local(i) = -1;
    }

    // Set materials used for thermodynamic initialisation
    int jj = 0;
    for (int ii = 0; ii < nic; ii++) {
        if (setup_config.thermo[ii].type == ThermodynamicsIC::Type::MATERIAL) {
            local(jj++) = setup_config.thermo[ii].value;
        }
    }

    // Find and set thermodynamic values for components from non-material based
    // thermodynamic input
    for (int imx = 0; imx < nmx; imx++) {
        int icp = mxfcp(imx);
        int iel = mxel(imx);
        for (int ii = 0; ii < mxncp(imx); ii++) {
            int imat = cpmat(icp);
            bool matching = false;
            for (int i = 0; i < nic; i++) {
                if (local(i) == imat) {
                    matching = true;
                    break;
                }
            }

            if (!matching) {
                switch (setup_config.materials[imat].type) {
                case Material::Type::REGION:
                case Material::Type::BACKGROUND:
                    cpdensity(icp) = eldensity(iel);
                    cpenergy(icp) = elenergy(iel);
                    break;

                default:
                    err.fail("ERROR: failed to rationalise thermodynamic IC "
                            "due to incorrect material type");
                    return;
                }
            }

            icp = cpnext(icp);
        }
    }
}

} // namespace kernel
} // namespace setup
} // namespace bookleaf
