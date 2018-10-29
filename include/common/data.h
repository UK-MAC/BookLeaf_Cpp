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
#ifndef BOOKLEAF_COMMON_DATA_H
#define BOOKLEAF_COMMON_DATA_H

#include <array>
#include <numeric>
#include <vector>

#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
#include <cuda.h>
#include <cuda_runtime.h>
#endif

#include "common/defs.h"
#include "common/data_id.h"
#include "common/view.h"



namespace bookleaf {

/**
 * \author  timrlaw
 *
 * This class manages a block of memory of given type and dimensions. When code
 * wants to read or update the data, it requests a View object, which handles
 * mapping indices to memory locations.
 *
 * This class is not templated as this prevents us from storing a collection of
 * instances in the DataControl class. To get around this, we use template
 * functions to allocate the data. Requests for views are templated separately,
 * which allows for additional flexibility (for example, if code wanted to view
 * some double-precision data as bytes, this is possible).
 */
class Data
{
public:
    typedef SizeType size_type;

public:
    Data() :
        id(DataID::COUNT),
        name(""),
        dims {0},
        len(0),
        host_ptr(nullptr),
        device_ptr(nullptr),
        allocated_T_size(0),
        typh_quant_id(-1)
    {
    }

    ~Data()
    {
        deallocate();
    }


    // -------------------------------------------------------------------------
    // Accessors
    // -------------------------------------------------------------------------
    /** \brief Has this been allocated? */
    bool isAllocated() const { return host_ptr != nullptr; }

    /** \brief Get the string name of the allocated type. */
    std::string getAllocatedType() const { return allocated_type; }

    /** \brief Return the ID. */
    DataID getId() const { return id; }

    /** \brief Return the name. */
    std::string getName() const { return name; }

    /** \brief Get the Typhon quant ID for this data. */
    int getTyphonHandle() const { return typh_quant_id; }

    /** \brief Return the size of the allocation. */
    size_type size() const { return len; }

    /** \brief Return the number of rows allocated. */
    size_type rows() const { return dims[0]; }

    /** \brief Return the number of columns allocated. */
    size_type cols() const { return dims[1]; }

    /** \brief Return a raw data pointer. */
    template <typename T> T *data() { return (T *) host_ptr; }

    /** \brief Return a const raw data pointer. */
    template <typename T> T const *data() const { return (T *) host_ptr; }


    // -------------------------------------------------------------------------
    // Views
    // -------------------------------------------------------------------------
    /** \brief Construct and return a view of the host allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    View<T, NumRows, NumCols>
    host();

    /** \brief Construct and return a view of the host allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    View<T, NumRows, NumCols>
    host(SizeType num_rows);

    /** \brief Construct and return a read-only view of the host allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    ConstView<T, NumRows, NumCols>
    chost() const;

    /** \brief Construct and return a read-only view of the host allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    ConstView<T, NumRows, NumCols>
    chost(SizeType num_rows) const;

    /** \brief Construct and return a view of the device allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    DeviceView<T, NumRows, NumCols>
    device();

    /** \brief Construct and return a view of the device allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    DeviceView<T, NumRows, NumCols>
    device(SizeType num_rows);

    /** \brief Construct and return a read-only view of the device allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    ConstDeviceView<T, NumRows, NumCols>
    cdevice() const;

    /** \brief Construct and return a read-only view of the device allocation. */
    template <typename T, SizeType NumRows, SizeType NumCols = 1>
    ConstDeviceView<T, NumRows, NumCols>
    cdevice(SizeType num_rows) const;


    // -------------------------------------------------------------------------
    // Mutators
    // -------------------------------------------------------------------------
    /** \brief Allocate memory for the data. */
    template <typename T>
    void
    allocate(
            T initial_value,
            DataID id,
            std::string _dname,
            size_type num_rows,
            size_type num_cols = 1);

    /** \brief Reallocate memory for the data. */
    template <typename T>
    void
    reallocate(
            T initial_value,
            size_type num_rows,
            size_type num_cols = 1);

    /** \brief Deallocate the data memory. */
    void
    deallocate();

    /** \brief Set the Typhon quant ID for this data. */
    void setTyphonHandle(int taddr) { typh_quant_id = taddr; }

    /** \brief Initialise partial sync data. */
    static void initPartialSync(
            int *send_indices,
            int *recv_indices,
            int nsend,
            int nrecv);

    /** \brief Clean up partial sync data. */
    static void killPartialSync();


    // -------------------------------------------------------------------------
    // Device sync
    // -------------------------------------------------------------------------
    /** \brief Copy the contents of device memory to host memory. */
    void syncHost(bool allow_partial = false) const;

    /** \brief Copy the contents of host memory to device memory. */
    void syncDevice(bool allow_partial = false) const;


private:
    static bool partial_sync;               //!< Partial sync enabled?

    static int      *host_sync_send_idx;    //!< Host partial sync send indices
    static int      *device_sync_send_idx;  //!< Device partial sync send indices
    static int      *host_sync_recv_idx;    //!< Host partial sync recv indices
    static int      *device_sync_recv_idx;  //!< Device partial sync recv indices
    static size_type sync_send_nidx;        //!< Number of sync send indices
    static size_type sync_recv_nidx;        //!< Number of sync recv indices

    static unsigned char *host_sync_buf;    //!< Host buffer for partial sync
    static unsigned char *device_sync_buf;  //!< Device buffer for partial sync
    static size_type      sync_buf_size;    //!< Partial sync buffer size (bytes)

    DataID         id;                //!< Global data ID
    std::string    name;              //!< Human-readable name

    size_type      dims[2];           //!< Dimensions
    size_type      len;               //!< Product of dimensions

    unsigned char *host_ptr;          //!< Allocated host memory
    unsigned char *device_ptr;        //!< Allocated device memory
    unsigned char *device_buf;        //!< Space for device sync
    size_type      allocated_T_size;  //!< Sizeof allocated type
    std::string    allocated_type;    //!< Name of allocated type

    int            typh_quant_id;     //!< Typhon quant ID
};



template <typename T, SizeType NumRows, SizeType NumCols>
View<T, NumRows, NumCols>
Data::host()
{
    if (!isAllocated()) {
        return View<T, NumRows, NumCols>(nullptr, 0);
    }

    return View<T, NumRows, NumCols>((T *) host_ptr, dims[0]);
}



template <typename T, SizeType NumRows, SizeType NumCols>
View<T, NumRows, NumCols>
Data::host(SizeType num_rows)
{
    if (!isAllocated()) {
        return View<T, NumRows, NumCols>(nullptr, 0);
    }

    return View<T, NumRows, NumCols>((T *) host_ptr, num_rows);
}



template <typename T, SizeType NumRows, SizeType NumCols>
ConstView<T, NumRows, NumCols>
Data::chost() const
{
    if (!isAllocated()) {
        return ConstView<T, NumRows, NumCols>(nullptr, 0);
    }

    return ConstView<T, NumRows, NumCols>((T *) host_ptr, dims[0]);
}



template <typename T, SizeType NumRows, SizeType NumCols>
ConstView<T, NumRows, NumCols>
Data::chost(SizeType num_rows) const
{
    if (!isAllocated()) {
        return ConstView<T, NumRows, NumCols>(nullptr, 0);
    }

    return ConstView<T, NumRows, NumCols>((T *) host_ptr, num_rows);
}



template <typename T, SizeType NumRows, SizeType NumCols>
DeviceView<T, NumRows, NumCols>
Data::device()
{
    if (!isAllocated()) {
        return DeviceView<T, NumRows, NumCols>(nullptr, 0);
    }

    return DeviceView<T, NumRows, NumCols>((T *) device_ptr, dims[0]);
}



template <typename T, SizeType NumRows, SizeType NumCols>
DeviceView<T, NumRows, NumCols>
Data::device(SizeType num_rows)
{
    if (!isAllocated()) {
        return DeviceView<T, NumRows, NumCols>(nullptr, 0);
    }

    return DeviceView<T, NumRows, NumCols>((T *) device_ptr, num_rows);
}



template <typename T, SizeType NumRows, SizeType NumCols>
ConstDeviceView<T, NumRows, NumCols>
Data::cdevice() const
{
    if (!isAllocated()) {
        return ConstDeviceView<T, NumRows, NumCols>(nullptr, 0);
    }

    return ConstDeviceView<T, NumRows, NumCols>((T *) device_ptr, dims[0]);
}



template <typename T, SizeType NumRows, SizeType NumCols>
ConstDeviceView<T, NumRows, NumCols>
Data::cdevice(SizeType num_rows) const
{
    if (!isAllocated()) {
        return ConstDeviceView<T, NumRows, NumCols>(nullptr, 0);
    }

    return ConstDeviceView<T, NumRows, NumCols>((T *) device_ptr, num_rows);
}



template <typename T>
void
Data::allocate(
        T initial_value,
        DataID id,
        std::string name,
        size_type num_rows,
        size_type num_cols)
{
    deallocate();

    // Set metadata
    this->id = id;
    this->name = name;
    dims[0] = num_rows;
    dims[1] = num_cols;

    allocated_T_size = sizeof(T);
    allocated_type = getTypeName<T>();

    // Allocate memory and fill with initial value
    len = num_rows * num_cols;

    host_ptr = (unsigned char *) malloc(len * sizeof(T));
    if (host_ptr == nullptr) {
        assert(false && "unhandled error");
    }

#ifdef BOOKLEAF_KOKKOS_CUDA_SUPPORT
    auto cuda_err = cudaMalloc((void **) &device_ptr, len * sizeof(T));
    if (cuda_err != cudaSuccess) {
        assert(false && "failed to allocate device memory");
    }

    device_buf = (unsigned char *) malloc(len * sizeof(T));
    if (device_buf == nullptr) {
        assert(false && "unhandled error");
    }
#else
    device_ptr = host_ptr;
    device_buf = nullptr;
#endif

    T *thost_ptr = (T *) host_ptr;
    std::fill(thost_ptr, thost_ptr + len, initial_value);

    syncDevice(false);
}



template <typename T>
void
Data::reallocate(
        T initial_value __attribute__((unused)),
        size_type num_rows __attribute__((unused)),
        size_type num_cols __attribute__((unused)))
{
    assert(false && "not yet implemented---do this when mixed cells working");
    assert(isAllocated() && "attempt to reallocate unallocated data");
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_DATA_H
