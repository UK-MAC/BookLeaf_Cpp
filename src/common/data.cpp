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

#include "common/constants.h"
#include "common/view.h"



namespace bookleaf {

using constants::NCORN;

namespace {

#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT

template <typename T>
void
gatherIndices(
        internal::KokkosDeviceView<T *>         dst,
        internal::KokkosDeviceView<T const *>   src,
        internal::KokkosDeviceView<int const *> indices,
        int nidx)
{
    Kokkos::parallel_for(
            RangePolicy(0, nidx),
            KOKKOS_LAMBDA (int const iidx)
    {
        dst(iidx) = src(indices(iidx));
    });
}



template <typename T>
void
gatherIndices(
        internal::KokkosDeviceView<T **>        dst,
        internal::KokkosDeviceView<T const **>  src,
        internal::KokkosDeviceView<int const *> indices,
        int nidx)
{
    Kokkos::parallel_for(
            RangePolicy(0, nidx),
            KOKKOS_LAMBDA (int const iidx)
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
        internal::KokkosHostView<T *>         dst,
        internal::KokkosHostView<T const *>   src,
        internal::KokkosHostView<int const *> indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        dst(iidx) = src(indices(iidx));
    }
}



template <typename T>
void
hostTransposeGatherIndices(
        typename internal::KokkosDeviceView<T **>::HostMirror dst,
        internal::KokkosHostView<T const **>                  src,
        internal::KokkosHostView<int const *>                 indices,
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
        internal::KokkosDeviceView<T *>         dst,
        internal::KokkosDeviceView<T const *>   src,
        internal::KokkosDeviceView<int const *> indices,
        int nidx)
{
    Kokkos::parallel_for(
            RangePolicy(0, nidx),
            KOKKOS_LAMBDA (int const iidx)
    {
        dst(indices(iidx)) = src(iidx);
    });
}



template <typename T>
void
scatterIndices(
        internal::KokkosDeviceView<T **>        dst,
        internal::KokkosDeviceView<T const **>  src,
        internal::KokkosDeviceView<int const *> indices,
        int nidx)
{
    Kokkos::parallel_for(
            RangePolicy(0, nidx),
            KOKKOS_LAMBDA (int const iidx)
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
        internal::KokkosHostView<T *>         dst,
        internal::KokkosHostView<T const *>   src,
        internal::KokkosHostView<int const *> indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        dst(indices(iidx)) = src(iidx);
    }
}



template <typename T>
void
hostTransposeScatterIndices(
        internal::KokkosHostView<T **>                              dst,
        typename internal::KokkosDeviceView<T const **>::HostMirror src,
        internal::KokkosHostView<int const *>                       indices,
        int nidx)
{
    for (int iidx = 0; iidx < nidx; iidx++) {
        int const idx = indices(iidx);
        for (int icn = 0; icn < NCORN; icn++) {
            dst(idx, icn) = src(iidx, icn);
        }
    }
}

#endif // BOOKLEAF_KOKKOS_CUDA_SUPPORT

template <typename T>
void
syncHostTranspose(
        T const *device_ptr,
        T *device_buf,
        T *host_ptr,
        Data::size_type const dims[2])
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    if (dims[1] == 1) {
        // Copy data directly from device
        internal::KokkosDeviceView<T *>
            device_mem((T *) device_ptr, dims[0]);
        internal::KokkosHostView<T *>
            host_mem((T *) host_ptr, dims[0]);

        Kokkos::deep_copy(host_mem, device_mem);

    } else {
        static_assert(
                !std::is_same<HostLayout, DeviceLayout>::value,
                "expecting a different device layout");

        // Copy data from device to scratch
        internal::KokkosDeviceView<T **>
            device_mem((T *) device_ptr, dims[0], dims[1]);
        typename internal::KokkosDeviceView<T **>::HostMirror
            mirror_mem((T *) device_buf, dims[0], dims[1]);

        Kokkos::deep_copy(mirror_mem, device_mem);

        // Transpose data
        internal::KokkosHostView<T **>
            host_mem((T *) host_ptr, dims[0], dims[1]);

        for (Data::size_type i = 0; i < dims[0]; i++) {
            for (Data::size_type j = 0; j < dims[1]; j++) {
                host_mem(i, j) = mirror_mem(i, j);
            }
        }
    }
#endif
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
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    internal::KokkosHostView<int const *>
        host_indices((int const *) _host_indices, nidx);
    internal::KokkosDeviceView<int const *>
        device_indices((int const *) _device_indices, nidx);

    if (dims[1] == 1) {
        internal::KokkosHostView<T *>
            host_arr((T *) _host_arr, dims[0]);
        internal::KokkosDeviceView<T const *>
            device_arr((T const *) _device_arr, dims[0]);

        internal::KokkosHostView<T *>
            host_buf((T *) _host_buf, nidx);
        internal::KokkosDeviceView<T *>
            device_buf((T *) _device_buf, nidx);

        // Gather data into buffer device-side
        gatherIndices(device_buf, device_arr, device_indices, nidx);

        // Copy buffer to host
        Kokkos::deep_copy(host_buf, device_buf);

        // Unpack host buffer into host array
        internal::KokkosHostView<T const *>
            chost_buf((T const *) _host_buf, nidx);
        hostScatterIndices(host_arr, chost_buf, host_indices, nidx);

    } else {
        internal::KokkosHostView<T **>
            host_arr((T *) _host_arr, dims[0], dims[1]);
        internal::KokkosDeviceView<T const **>
            device_arr((T const *) _device_arr, dims[0], dims[1]);

        internal::KokkosHostView<T **>
            host_buf((T *) _host_buf, nidx, dims[1]);
        internal::KokkosDeviceView<T **>
            device_buf((T *) _device_buf, nidx, dims[1]);

        // Gather data into buffer device-side
        gatherIndices(device_buf, device_arr, device_indices, nidx);

        // Copy buffer to host
        typename internal::KokkosDeviceView<T **>::HostMirror
            host_buf_mirror((T *) _host_buf, nidx, dims[1]);

        Kokkos::deep_copy(host_buf_mirror, device_buf);

        // Unpack host buffer into host array
        typename internal::KokkosDeviceView<T const **>::HostMirror
            chost_buf_mirror((T *) _host_buf, nidx, dims[1]);
        hostTransposeScatterIndices(host_arr, chost_buf_mirror, host_indices, nidx);
    }
#endif
}



template <typename T>
void
syncDeviceTranspose(
        T *device_ptr,
        T *device_buf,
        T const *host_ptr,
        Data::size_type const dims[2])
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    if (dims[1] == 1) {
        // Copy data directly to device
        internal::KokkosHostView<T *>
            host_mem((T *) host_ptr, dims[0]);
        internal::KokkosDeviceView<T *>
            device_mem((T *) device_ptr, dims[0]);

        Kokkos::deep_copy(device_mem, host_mem);

    } else {
        static_assert(
                !std::is_same<HostLayout, DeviceLayout>::value,
                "expecting a different device layout");

        // Transpose data to scratch
        internal::KokkosHostView<T **>
            host_mem((T *) host_ptr, dims[0], dims[1]);
        typename internal::KokkosDeviceView<T **>::HostMirror
            mirror_mem((T *) device_buf, dims[0], dims[1]);

        for (Data::size_type i = 0; i < dims[0]; i++) {
            for (Data::size_type j = 0; j < dims[1]; j++) {
                mirror_mem(i, j) = host_mem(i, j);
            }
        }

        // Copy scratch to device
        internal::KokkosDeviceView<T **>
            device_mem((T *) device_ptr, dims[0], dims[1]);

        Kokkos::deep_copy(device_mem, mirror_mem);
    }
#endif
}



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
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    internal::KokkosHostView<int const *>
        host_indices((int const *) _host_indices, nidx);
    internal::KokkosDeviceView<int const *>
        device_indices((int const *) _device_indices, nidx);

    if (dims[1] == 1) {
        internal::KokkosHostView<T const *>
            host_arr((T const *) _host_arr, dims[0]);
        internal::KokkosDeviceView<T *>
            device_arr((T *) _device_arr, dims[0]);

        internal::KokkosHostView<T *>
            host_buf((T *) _host_buf, nidx);
        internal::KokkosDeviceView<T *>
            device_buf((T *) _device_buf, nidx);

        // Gather data into buffer host-side
        hostGatherIndices(host_buf, host_arr, host_indices, nidx);

        // Copy buffer to device
        Kokkos::deep_copy(device_buf, host_buf);

        // Unpack device buffer into device array
        internal::KokkosDeviceView<T const *>
            cdevice_buf((T const *) _device_buf, nidx);
        scatterIndices(device_arr, cdevice_buf, device_indices, nidx);

    } else {
        internal::KokkosHostView<T const **>
            host_arr((T const *) _host_arr, dims[0], dims[1]);
        internal::KokkosDeviceView<T **>
            device_arr((T *) _device_arr, dims[0], dims[1]);

        typename internal::KokkosDeviceView<T **>::HostMirror
            host_buf_mirror((T *) _host_buf, nidx, dims[1]);
        internal::KokkosDeviceView<T **>
            device_buf((T *) _device_buf, nidx, dims[1]);

        // Gather data into buffer host-side
        hostTransposeGatherIndices(host_buf_mirror, host_arr, host_indices, nidx);

        // Copy buffer to device
        Kokkos::deep_copy(device_buf, host_buf_mirror);

        // Unpack device buffer into device array
        typename internal::KokkosDeviceView<T const **>
            cdevice_buf((T const *) _device_buf, nidx, dims[1]);
        scatterIndices(device_arr, cdevice_buf, device_indices, nidx);
    }
#endif
}

} // namespace internal

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

void
Data::deallocate()
{
    if (isAllocated()) {

#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
        auto cuda_err = cudaFree(device_ptr);
        if (cuda_err != cudaSuccess) {
            assert(false && "failed to free device memory");
        }

        free(device_buf);
#endif
        device_ptr = nullptr;
        device_buf = nullptr;

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
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    assert(!partial_sync);

    cudaError_t cuda_err;

    // Host indices
    sync_send_nidx = nsend;
    sync_recv_nidx = nrecv;

    host_sync_send_idx = new int[nsend];
    host_sync_recv_idx = new int[nrecv];

    std::copy(send_indices, send_indices + nsend, host_sync_send_idx);
    std::copy(recv_indices, recv_indices + nrecv, host_sync_recv_idx);

    // Device indices
    cuda_err = cudaMalloc(&device_sync_send_idx, sizeof(int) * nsend);
    cuda_err = cudaMalloc(&device_sync_recv_idx, sizeof(int) * nrecv);

    cuda_err = cudaMemcpy(device_sync_send_idx, host_sync_send_idx,
            sizeof(int) * nsend, cudaMemcpyHostToDevice);
    cuda_err = cudaMemcpy(device_sync_recv_idx, host_sync_recv_idx,
            sizeof(int) * nrecv, cudaMemcpyHostToDevice);

    // Host buffer
    sync_buf_size = sizeof(double) * NCORN * std::max(nsend, nrecv);

    host_sync_buf = new unsigned char[sync_buf_size];

    // Device buffer
    cuda_err = cudaMalloc(&device_sync_buf, sync_buf_size);

    partial_sync = true;
#endif
}



void
Data::killPartialSync()
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    if (partial_sync) {
        cudaError_t cuda_err;

        delete[] host_sync_send_idx;
        delete[] host_sync_recv_idx;
        delete[] host_sync_buf;

        cuda_err = cudaFree(device_sync_send_idx);
        cuda_err = cudaFree(device_sync_recv_idx);
        cuda_err = cudaFree(device_sync_buf);

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
#endif
}



void
Data::syncHost(bool allow_partial) const
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
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
            assert(false && "unsupported type");
        }

    } else {
        if (allocated_type == "double") {
            syncHostTranspose<double>(
                    (double *) device_ptr,
                    (double *) device_buf,
                    (double *) host_ptr,
                    dims);

        } else if (allocated_type == "integer") {
            syncHostTranspose<int>(
                    (int *) device_ptr,
                    (int *) device_buf,
                    (int *) host_ptr,
                    dims);

        } else if (allocated_type == "boolean") {
            syncHostTranspose<unsigned char>(
                    (unsigned char *) device_ptr,
                    (unsigned char *) device_buf,
                    (unsigned char *) host_ptr,
                    dims);

        } else {
            assert(false && "unsupported type");
        }
    }
#endif
}



void
Data::syncDevice(bool allow_partial) const
{
#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
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
            assert(false && "unsupported type");
        }

    } else {
        if (allocated_type == "double") {
            syncDeviceTranspose<double>(
                    (double *) device_ptr,
                    (double *) device_buf,
                    (double *) host_ptr,
                    dims);

        } else if (allocated_type == "integer") {
            syncDeviceTranspose<int>(
                    (int *) device_ptr,
                    (int *) device_buf,
                    (int *) host_ptr,
                    dims);

        } else if (allocated_type == "boolean") {
            syncDeviceTranspose<unsigned char>(
                    (unsigned char *) device_ptr,
                    (unsigned char *) device_buf,
                    (unsigned char *) host_ptr,
                    dims);

        } else {
            assert(false && "unsupported type");
        }
    }
#endif
}

} // namespace bookleaf
