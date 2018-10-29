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
#ifndef BOOKLEAF_COMMON_CUDA_UTILS_H
#define BOOKLEAF_COMMON_CUDA_UTILS_H

#include <functional>
#include <cassert>
#include <iostream>

#include <cuda.h>
#include <cuda_runtime.h>

// Taken these max and min double values from a CUDA library
#define NPP_MAXABS_64F ( 1.7976931348623158e+308 )
#define NPP_MINABS_64F ( 2.2250738585072014e-308 )



namespace bookleaf {
namespace internal {

template <typename F>
void __global__
dispatchIteration(int len, F f)
{
    int const idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < len) f(idx);
}

} // namespace internal

template <typename F>
void
dispatchCuda(int len, F f)
{
    int min_grid_size = 0;
    int block_size = 1024;
    cudaOccupancyMaxPotentialBlockSize(&min_grid_size, &block_size,
            internal::dispatchIteration<F>, 0, 0);

    dim3 grid;
    grid.x = (len + block_size - 1) / block_size;
    grid.y = 1;
    grid.z = 1;

    dim3 block;
    block.x = block_size;
    block.y = 1;
    block.z = 1;

    // Wait for previous kernel to finish before launching the new one
    auto cuda_err = cudaDeviceSynchronize();
    if (cuda_err != cudaSuccess) {
        assert(false);
    }

    internal::dispatchIteration<<<grid, block>>>(len, f);
}



inline void
cudaSync()
{
    // Can comment this out to slightly increase overlap, at the expense of
    // timers being wrong. Haven't observed a big performance difference.
    cudaDeviceSynchronize();
}



inline void
cudaAssert(cudaError_t err)
{
    if (err != cudaSuccess) {
        assert(false && "unhandled cuda error");
    }
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_CUDA_UTILS_H
