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
#ifndef BOOKLEAF_UTILITIES_DATA_COPY_H
#define BOOKLEAF_UTILITIES_DATA_COPY_H

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/view.h"
#include "common/cuda_utils.h"



namespace bookleaf {
namespace utils {
namespace kernel {

using constants::NCORN;

template <typename T>
void
copy(
        DeviceView<T, VarDim>      dst,
        ConstDeviceView<T, VarDim> src,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    Kokkos::parallel_for(
            RangePolicy(0, len),
            KOKKOS_LAMBDA (int const i)
    {
        dst(i) = src(i);
    });

    cudaSync();
}



template <typename T>
void
copy(
        DeviceView<T, VarDim, NCORN>      dst,
        ConstDeviceView<T, VarDim, NCORN> src,
        int len)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    Kokkos::parallel_for(
            RangePolicy(0, len),
            KOKKOS_LAMBDA (int const i)
    {
        for (int icn = 0; icn < NCORN; icn++) {
            dst(i, icn) = src(i, icn);
        }
    });

    cudaSync();
}

} // namespace kernel
} // namespace utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DATA_COPY_H
