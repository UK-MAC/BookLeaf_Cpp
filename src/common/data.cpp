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
#include "common/data.h"

#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "common/cuda_utils.h"



namespace bookleaf {

using constants::NCORN;

bool            Data::partial_sync         = false;
int            *Data::host_sync_send_idx   = nullptr;
int            *Data::device_sync_send_idx = nullptr;
int            *Data::host_sync_recv_idx   = nullptr;
int            *Data::device_sync_recv_idx = nullptr;
Data::size_type Data::sync_send_nidx       = 0;
Data::size_type Data::sync_recv_nidx       = 0;
unsigned char  *Data::host_sync_buf        = nullptr;
unsigned char  *Data::device_sync_buf      = nullptr;
Data::size_type Data::sync_buf_size        = 0;
unsigned char  *Data::transpose_buf        = nullptr;
Data::size_type Data::transpose_size       = 0;

namespace {

template <typename T>
void
gatherIndices(
        DeviceView<T, VarDim>        dst,
        ConstDeviceView<T, VarDim>   src,
        ConstDeviceView<int, VarDim> indices,
        int nidx)
{
    dispatchCuda(
            nidx,
            [=] __device__ (int const iidx)
    {
        dst(iidx) = src(indices(iidx));
    });
}



template <typename T>
void
gatherIndices(
        DeviceView<T, VarDim, NCORN>      dst,
        ConstDeviceView<T, VarDim, NCORN> src,
        ConstDeviceView<int, VarDim>      indices,
        int nidx)
{
    dispatchCuda(
            nidx,
            [=] __device__ (int const iidx)
    {
        int const idx = indices(iidx);
        for (int icn = 0; icn < NCORN; icn++) {
            dst(iidx, icn) = src(idx, icn);
        }
    });
}



template <typename T>
void
hostGatherIndices(
        View<T, VarDim>        dst,
        ConstView<T, VarDim>   src,
        ConstView<int, VarDim> indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        dst(iidx) = src(indices(iidx));
    }
}



template <typename T>
void
hostGatherIndices(
        DeviceView<T, VarDim, NCORN> dst,
        ConstView<T, VarDim, NCORN>  src,
        ConstView<int, VarDim>       indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        int const idx = indices(iidx);
        for (int icn = 0; icn < NCORN; icn++) {
            dst(iidx, icn) = src(idx, icn);
        }
    }
}



template <typename T>
void
scatterIndices(
        DeviceView<T, VarDim>        dst,
        ConstDeviceView<T, VarDim>   src,
        ConstDeviceView<int, VarDim> indices,
        int nidx)
{
    dispatchCuda(
            nidx,
            [=] __device__ (int const iidx)
    {
        dst(indices(iidx)) = src(iidx);
    });
}



template <typename T>
void
scatterIndices(
        DeviceView<T, VarDim, NCORN>      dst,
        ConstDeviceView<T, VarDim, NCORN> src,
        ConstDeviceView<int, VarDim>      indices,
        int nidx)
{
    dispatchCuda(
            nidx,
            [=] __device__ (int const iidx)
    {
        int const idx = indices(iidx);
        for (int icn = 0; icn < NCORN; icn++) {
            dst(idx, icn) = src(iidx, icn);
        }
    });
}



template <typename T>
void
hostScatterIndices(
        View<T, VarDim>        dst,
        ConstView<T, VarDim>   src,
        ConstView<int, VarDim> indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        dst(indices(iidx)) = src(iidx);
    }
}



template <typename T>
void
hostScatterIndices(
        View<T, VarDim, NCORN>            dst,
        ConstDeviceView<T, VarDim, NCORN> src,
        ConstView<int, VarDim>            indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        int const idx = indices(iidx);
        for (int icn = 0; icn < NCORN; icn++) {
            dst(idx, icn) = src(iidx, icn);
        }
    }
}



template <typename T>
void
syncHostTranspose(
        T const *_device_arr,
        T       *_host_arr,
        T       *_transpose_buf,
        Data::size_type const dims[2])
{
    Data::size_type const len = dims[0] * dims[1];

    if (dims[1] == 1) {
        // Copy data directly from device
        auto cuda_err = cudaMemcpy(_host_arr, _device_arr, len * sizeof(T),
                cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to sync host");
        }

    } else {
        // Copy data from device to scratch
        auto cuda_err = cudaMemcpy(_transpose_buf, _device_arr,
                len * sizeof(T), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to sync host");
        }

        // Transpose data
        View<T, VarDim, NCORN> host_arr(_host_arr, dims[0]);
        ConstDeviceView<T, VarDim, NCORN> transpose_buf(_transpose_buf, dims[0]);

        for (Data::size_type i = 0; i < dims[0]; i++) {
            for (Data::size_type j = 0; j < dims[1]; j++) {
                host_arr(i, j) = transpose_buf(i, j);
            }
        }
    }
}



template <typename T>
void
partialSyncHost(
        T   const *_device_arr,
        int const *_host_indices,
        int const *_device_indices,
        T         *_host_arr,
        T         *_host_buf,
        T         *_device_buf,
        Data::size_type const dims[2],
        int nidx)
{
    if (dims[1] == 1) {
        ConstView<int, VarDim> hst_indices(_host_indices, nidx);
        ConstDeviceView<int, VarDim> dev_indices(_device_indices, nidx);

        View<T, VarDim> hst_arr(_host_arr, dims[0]);
        ConstDeviceView<T, VarDim> dev_arr(_device_arr, dims[0]);

        View<T, VarDim> hst_buf(_host_buf, nidx);
        DeviceView<T, VarDim> dev_buf(_device_buf, nidx);

        // Gather data into buffer device-side
        gatherIndices<T>(dev_buf, dev_arr, dev_indices, nidx);

        // Copy buffer to host
        auto cuda_err = cudaMemcpy(hst_buf.data(), dev_buf.data(),
                nidx * sizeof(T), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to copy device to host");
        }

        // Unpack host buffer into host array
        ConstView<T, VarDim> chst_buf(hst_buf);
        hostScatterIndices<T>(hst_arr, chst_buf, hst_indices, nidx);

    } else {
        ConstView<int, VarDim> hst_indices(_host_indices, nidx);
        ConstDeviceView<int, VarDim> dev_indices(_device_indices, nidx);

        View<T, VarDim, NCORN> hst_arr(_host_arr, dims[0]);
        ConstDeviceView<T, VarDim, NCORN> dev_arr(_device_arr, dims[0]);

        DeviceView<T, VarDim, NCORN> dev_buf(_device_buf, nidx);

        // Gather data into buffer device-side
        gatherIndices<T>(dev_buf, dev_arr, dev_indices, nidx);

        // Copy buffer to host
        auto cuda_err = cudaMemcpy(_host_buf, _device_buf,
                nidx * NCORN * sizeof(T), cudaMemcpyDeviceToHost);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to copy device to host");
        }

        // Unpack host buffer into host array
        ConstDeviceView<T, VarDim, NCORN> hst_buf(_host_buf, nidx);
        hostScatterIndices<T>(hst_arr, hst_buf, hst_indices, nidx);
    }
}



template <typename T>
void
syncDeviceTranspose(
        T const *_host_arr,
        T       *_device_arr,
        T       *_transpose_buf,
        Data::size_type const dims[2])
{
    Data::size_type const len = dims[0] * dims[1];

    if (dims[1] == 1) {
        // Copy data directly from host
        auto cuda_err = cudaMemcpy(_device_arr, _host_arr,
                len * sizeof(T), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to sync device");
        }

    } else {
        // Transpose data
        ConstView<T, VarDim, NCORN> host_arr(_host_arr, dims[0]);
        DeviceView<T, VarDim, NCORN> transpose_buf(_transpose_buf, dims[0]);

        for (Data::size_type i = 0; i < dims[0]; i++) {
            for (Data::size_type j = 0; j < dims[1]; j++) {
                transpose_buf(i, j) = host_arr(i, j);
            }
        }

        // Copy data from scratch to device
        auto cuda_err = cudaMemcpy(_device_arr, _transpose_buf,
                len * sizeof(T), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to sync device");
        }
    }
}



//template <typename T, SizeType NumCols>
//void
//partialSyncDevice(
        //ConstView<T, VarDim, NumCols> host_arr,
        //ConstView<int, VarDim>        host_indices,
        //ConstView<int, VarDim>        device_indices,
        //View<T, VarDim, NumCols>      device_arr,
        //View<T, VarDim, NumCols>      host_buf,
        //View<T, VarDim, NumCols>      device_buf,
        //int nidx)
//{
    //// Gather data into buffer host-side
    //hostGatherIndices(host_buf, host_arr, host_indices, nidx);
    //auto cuda_err = cudaDeviceSynchronize();
    //if (cuda_err != cudaSuccess) {
        //assert(false && "failed to gather indices");
    //}

    //// Copy buffer to device
    //cuda_err = cudaMemcpy(device_buf.data(), host_buf.data(),
            //nidx * NumCols * sizeof(T), cudaMemcpyHostToDevice);
    //if (cuda_err != cudaSuccess) {
        //assert(false && "failed to copy host to device");
    //}

    //// Unpack device buffer into device array
    //ConstView<T, VarDim, NumCols> cdevice_buf(device_buf);
    //scatterIndices(device_arr, cdevice_buf, device_indices, nidx);
//}
template <typename T>
void
partialSyncDevice(
        T   const *_host_arr,
        int const *_host_indices,
        int const *_device_indices,
        T         *_device_arr,
        T         *_host_buf,
        T         *_device_buf,
        Data::size_type const dims[2],
        int nidx)
{
    if (dims[1] == 1) {
        ConstView<int, VarDim> hst_indices(_host_indices, nidx);
        ConstDeviceView<int, VarDim> dev_indices(_device_indices, nidx);

        ConstView<T, VarDim> hst_arr(_host_arr, dims[0]);
        DeviceView<T, VarDim> dev_arr(_device_arr, dims[0]);

        View<T, VarDim> hst_buf(_host_buf, nidx);
        DeviceView<T, VarDim> dev_buf(_device_buf, nidx);

        // Gather data into buffer host-side
        hostGatherIndices<T>(hst_buf, hst_arr, hst_indices, nidx);

        // Copy buffer to device
        auto cuda_err = cudaMemcpy(dev_buf.data(), hst_buf.data(),
                nidx * sizeof(T), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to copy host to device");
        }

        // Unpack device buffer into device array
        ConstDeviceView<T, VarDim> cdev_buf(dev_buf);
        scatterIndices<T>(dev_arr, cdev_buf, dev_indices, nidx);

    } else {
        ConstView<int, VarDim> hst_indices(_host_indices, nidx);
        ConstDeviceView<int, VarDim> dev_indices(_device_indices, nidx);

        ConstView<T, VarDim, NCORN> hst_arr(_host_arr, dims[0]);
        DeviceView<T, VarDim, NCORN> dev_arr(_device_arr, dims[0]);

        DeviceView<T, VarDim, NCORN> hst_buf(_host_buf, nidx);
        DeviceView<T, VarDim, NCORN> dev_buf(_device_buf, nidx);

        // Gather data into buffer host-side
        hostGatherIndices<T>(hst_buf, hst_arr, hst_indices, nidx);

        // Copy buffer to device
        auto cuda_err = cudaMemcpy(dev_buf.data(), hst_buf.data(),
                nidx * NCORN * sizeof(T), cudaMemcpyHostToDevice);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to copy host to device");
        }

        // Unpack device buffer into device array
        ConstDeviceView<T, VarDim, NCORN> cdev_buf(dev_buf);
        scatterIndices<T>(dev_arr, cdev_buf, dev_indices, nidx);
    }
}

} // namespace

void
Data::deallocate()
{
    if (isAllocated()) {

        auto cuda_err = cudaFree(device_ptr);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to free device memory");
        }
        device_ptr = nullptr;

        free(host_ptr);
        host_ptr = nullptr;

        dims[0] = 0;
        dims[1] = 0;

        typh_quant_id = -1;
        allocated_T_size = 0;
        len = 0;
        name = "";
        id = DataID::COUNT;
    }
}



void
Data::initPartialSync(
        int *send_indices,
        int *recv_indices,
        int nsend,
        int nrecv)
{
    assert(!partial_sync);

    // Host indices
    sync_send_nidx = nsend;
    sync_recv_nidx = nrecv;

    host_sync_send_idx = new int[nsend];
    host_sync_recv_idx = new int[nrecv];

    std::copy(send_indices, send_indices + nsend, host_sync_send_idx);
    std::copy(recv_indices, recv_indices + nrecv, host_sync_recv_idx);

    // Device indices
    cudaAssert(cudaMalloc(&device_sync_send_idx, sizeof(int) * nsend));
    cudaAssert(cudaMalloc(&device_sync_recv_idx, sizeof(int) * nrecv));

    cudaAssert(cudaMemcpy(device_sync_send_idx, host_sync_send_idx,
            sizeof(int) * nsend, cudaMemcpyHostToDevice));
    cudaAssert(cudaMemcpy(device_sync_recv_idx, host_sync_recv_idx,
            sizeof(int) * nrecv, cudaMemcpyHostToDevice));

    // Host buffer
    sync_buf_size = sizeof(double) * NCORN * std::max(nsend, nrecv);

    host_sync_buf = new unsigned char[sync_buf_size];

    // Device buffer
    cudaAssert(cudaMalloc(&device_sync_buf, sync_buf_size));

    partial_sync = true;
}



void
Data::killPartialSync()
{
    if (partial_sync) {
        delete[] host_sync_send_idx;
        delete[] host_sync_recv_idx;
        delete[] host_sync_buf;

        cudaAssert(cudaFree(device_sync_send_idx));
        cudaAssert(cudaFree(device_sync_recv_idx));
        cudaAssert(cudaFree(device_sync_buf));

        host_sync_send_idx   = nullptr;
        device_sync_send_idx = nullptr;
        host_sync_recv_idx   = nullptr;
        device_sync_recv_idx = nullptr;
        sync_send_nidx       = 0;
        sync_recv_nidx       = 0;
        host_sync_buf        = nullptr;
        device_sync_buf      = nullptr;
        sync_buf_size        = 0;

        partial_sync = false;
    }

    if (transpose_size > 0) {
        free(transpose_buf);
        transpose_buf = nullptr;
        transpose_size = 0;
    }
}



void
Data::syncHost(bool allow_partial) const
{
    if (partial_sync && allow_partial) {
        if (allocated_type == "double") {
            partialSyncHost<double>(
                    (double *) device_ptr,
                    (int *) host_sync_send_idx,
                    (int *) device_sync_send_idx,
                    (double *) host_ptr,
                    (double *) host_sync_buf,
                    (double *) device_sync_buf,
                    dims,
                    sync_send_nidx);

        } else if (allocated_type == "integer") {
            partialSyncHost<int>(
                    (int *) device_ptr,
                    (int *) host_sync_send_idx,
                    (int *) device_sync_send_idx,
                    (int *) host_ptr,
                    (int *) host_sync_buf,
                    (int *) device_sync_buf,
                    dims,
                    sync_send_nidx);

        } else {
            assert(false && "invalid sync type");
        }

    } else {
        if (allocated_type == "double") {
            syncHostTranspose<double>(
                    (double *) device_ptr,
                    (double *) host_ptr,
                    (double *) transpose_buf,
                    dims);

        } else if (allocated_type == "integer") {
            syncHostTranspose<int>(
                    (int *) device_ptr,
                    (int *) host_ptr,
                    (int *) transpose_buf,
                    dims);

        } else {
            assert(false && "invalid sync type");
        }
    }
}



void
Data::syncDevice(bool allow_partial) const
{
    if (partial_sync && allow_partial) {
        if (allocated_type == "double") {
            partialSyncDevice<double>(
                    (double *) host_ptr,
                    (int *) host_sync_recv_idx,
                    (int *) device_sync_recv_idx,
                    (double *) device_ptr,
                    (double *) host_sync_buf,
                    (double *) device_sync_buf,
                    dims,
                    sync_recv_nidx);

        } else if (allocated_type == "integer") {
            partialSyncDevice<int>(
                    (int *) host_ptr,
                    (int *) host_sync_recv_idx,
                    (int *) device_sync_recv_idx,
                    (int *) device_ptr,
                    (int *) host_sync_buf,
                    (int *) device_sync_buf,
                    dims,
                    sync_recv_nidx);

        } else {
            assert(false && "invalid sync type");
        }

    } else {
        if (allocated_type == "double") {
            syncDeviceTranspose<double>(
                    (double *) host_ptr,
                    (double *) device_ptr,
                    (double *) transpose_buf,
                    dims);

        } else if (allocated_type == "integer") {
            syncDeviceTranspose<int>(
                    (int *) host_ptr,
                    (int *) device_ptr,
                    (int *) transpose_buf,
                    dims);

        } else {
            assert(false && "invalid sync type");
        }
    }
}

} // namespace bookleaf
