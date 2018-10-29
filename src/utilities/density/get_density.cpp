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
#include "utilities/density/get_density.h"

#include <algorithm>
#include <numeric>
#include <functional>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/sizes.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace density {
namespace kernel {

void
getDensity(
        ConstDeviceView<double, VarDim> mass,
        ConstDeviceView<double, VarDim> volume,
        DeviceView<double, VarDim>      density,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    // Mass conserved, get density from updated volume
    Kokkos::parallel_for(
            RangePolicy(0, len),
            KOKKOS_LAMBDA (int const i)
    {
        density(i) = mass(i) / volume(i);
    });

    cudaSync();
}

} // namespace kernel

namespace driver {

void
getDensity(Sizes const &sizes, DataControl &data)
{
    int const nel = sizes.nel;

    auto elmass    = data[DataID::ELMASS].cdevice<double, VarDim>();
    auto elvolume  = data[DataID::ELVOLUME].cdevice<double, VarDim>();
    auto eldensity = data[DataID::ELDENSITY].device<double, VarDim>();

    kernel::getDensity(elmass, elvolume, eldensity, nel);

    int const ncp = sizes.ncp;
    if (ncp > 0) {
        auto cpmass    = data[DataID::CPMASS].cdevice<double, VarDim>();
        auto cpvolume  = data[DataID::CPVOLUME].cdevice<double, VarDim>();
        auto cpdensity = data[DataID::CPDENSITY].device<double, VarDim>();

        kernel::getDensity(cpmass, cpvolume, cpdensity, ncp);
    }
}

} // namespace driver
} // namespace density
} // namespace bookleaf
