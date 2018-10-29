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
#include "packages/hydro/driver/get_cs2.h"

#include <Kokkos_Core.hpp>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/sizes.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"
#include "utilities/misc/average.h"



namespace bookleaf {
namespace hydro {
namespace kernel {

void
getCs2(
        ConstDeviceView<double, VarDim> elvisc,
        DeviceView<double, VarDim>      elcs2,
        int nel)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Apply Q correction to soundspeed^2 (cs^2 = cs_eos^2 + 2Q/rho)
    Kokkos::parallel_for(
            RangePolicy(0, nel),
            KOKKOS_LAMBDA (int const iel)
    {
        elcs2(iel) += elvisc(iel);
    });

    cudaSync();
}

} // namespace kernel

namespace driver {

void
getCs2(
        Sizes const &sizes,
        DataControl &data)
{
    auto elcs2  = data[DataID::ELCS2].device<double, VarDim>();
    auto elvisc = data[DataID::ELVISC].cdevice<double, VarDim>();

    kernel::getCs2(elvisc, elcs2, sizes.nel);

    if (sizes.ncp > 0) {
        utils::driver::average(sizes, DataID::FRMASS, DataID::CPCS2,
                DataID::CPVISC, DataID::ELCS2, data);
    }
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
