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
#include "packages/ale/kernel/get_dt.h"

#include <limits>
#include <cmath>

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include <cub/device/device_reduce.cuh>

#include "common/constants.h"
#include "common/data_control.h"
#include "common/cuda_utils.h"
#include "packages/ale/config.h"



namespace bookleaf {
namespace ale {
namespace kernel {

void
getDt(
        ale::Config const &ale,
        int nel,
        double zerocut,
        double ale_sf,
        bool zeul,
        ConstDeviceView<double, VarDim, NCORN> cnu,
        ConstDeviceView<double, VarDim, NCORN> cnv,
        ConstDeviceView<double, VarDim>        ellength,
        DeviceView<double, VarDim>             scratch,
        double &rdt,
        int &idt,
        std::string &sdt)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    double w2 = std::numeric_limits<double>::max();
    int ii = 0;

    if (zeul) {
        dispatchCuda(
                nel,
                [=] __device__ (int const iel)
        {
            // Minimise node velocity squared
            double w1 = -NPP_MAXABS_64F;
            for (int icn = 0; icn < NCORN; icn++) {
                double w = cnu(iel, icn) * cnu(iel, icn) +
                           cnv(iel, icn) * cnv(iel, icn);
                w1 = max(w1, w);
            }

            scratch(iel) = ellength(iel) / max(w1, zerocut);
        });

        cudaDeviceSynchronize();

        auto &cub_storage_len = const_cast<SizeType &>(ale.cub_storage_len);
        auto cuda_err = cub::DeviceReduce::ArgMin(
                ale.cub_storage,
                cub_storage_len,
                scratch.data(),
                ale.cub_out,
                nel);

        cudaDeviceSynchronize();

        cub::KeyValuePair<int, double> res;
        cudaMemcpy(&res, ale.cub_out, sizeof(cub::KeyValuePair<int, double>),
                cudaMemcpyDeviceToHost);

        w2 = res.value;
        ii = res.key;

    } else {
        // XXX Missing code that can't (or hasn't) been merged
    }

    rdt = ale_sf*sqrt(w2);
    idt = ii;
    sdt = "     ALE";
}

} // namespace kernel
} // namespace ale
} // namespace bookleaf
