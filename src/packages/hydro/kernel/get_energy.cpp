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
#include "packages/hydro/kernel/get_energy.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
getEnergy(
        double dt,
        double zerocut,
        ConstDeviceView<double, VarDim, NCORN> cnfx,
        ConstDeviceView<double, VarDim, NCORN> cnfy,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        ConstDeviceView<double, VarDim>        elmass,
        DeviceView<double, VarDim>             elenergy,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // FdS internal energy update
    dispatchCuda(
            nel,
            [=] __device__ (int const iel)
    {
        double w1 = cnfx(iel, 0) * cnu(iel, 0) +
                    cnfy(iel, 0) * cnv(iel, 0) +
                    cnfx(iel, 1) * cnu(iel, 1) +
                    cnfy(iel, 1) * cnv(iel, 1) +
                    cnfx(iel, 2) * cnu(iel, 2) +
                    cnfy(iel, 2) * cnv(iel, 2) +
                    cnfx(iel, 3) * cnu(iel, 3) +
                    cnfy(iel, 3) * cnv(iel, 3);

        w1 = -w1 / max(elmass(iel), zerocut);
        elenergy(iel) += w1 * dt;
    });

    cudaSync();
}

} // namespace kernel
} // namespace hydro
} // namespace bookleaf
