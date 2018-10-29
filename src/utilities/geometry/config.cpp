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
#include "utilities/geometry/config.h"

#include <cub/cub.cuh>

#include "common/sizes.h"
#include "common/error.h"



namespace bookleaf {
namespace geometry {

void
initGeometryConfig(
        Sizes const &sizes,
        geometry::Config &geom,
        Error &err)
{
    // Init Cub reductions
    int *tmp_in;
    auto cuda_err = cudaMalloc(&tmp_in, sizeof(int) * sizes.nel);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Failed to allocate device memory");
        return;
    }

    cuda_err = cudaMalloc(&geom.cub_out, sizeof(int));
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Failed to allocate device memory");
        return;
    }

    geom.cub_storage = nullptr;
    geom.cub_storage_len = 0;

    cuda_err = cub::DeviceReduce::Min(
            geom.cub_storage,
            geom.cub_storage_len,
            tmp_in,
            geom.cub_out,
            sizes.nel);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Failed to calculate reduction scratch");
        return;
    }

    cuda_err = cudaMalloc(&geom.cub_storage, geom.cub_storage_len);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Failed to allocate device memory");
        return;
    }

    cuda_err = cudaFree(tmp_in);
    if (cuda_err != cudaSuccess) {
        FAIL_WITH_LINE(err, "ERROR: Failed to free device memory");
        return;
    }
}



void
killGeometryConfig(
        geometry::Config &geom)
{
    if (geom.cub_storage != nullptr) {
        cudaFree(geom.cub_out);
        cudaFree(geom.cub_storage);
        geom.cub_storage = nullptr;
        geom.cub_storage_len = 0;
    }
}

} // namespace geometry
} // namespace bookleaf
