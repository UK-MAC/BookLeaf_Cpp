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
#ifndef BOOKLEAF_COMMON_DATA_CONTROL_H
#define BOOKLEAF_COMMON_DATA_CONTROL_H

#include <string>
#include <vector>
#include <cassert>
#include <type_traits>

#ifdef BOOKLEAF_DEBUG
#include <iostream>
#endif

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/data.h"
#include "common/view.h"
#include "common/data_id.h"



namespace bookleaf {

struct Config;
struct Error;
struct Sizes;

class DataControl
{
public:
    typedef SizeType size_type;

private:
    static int           constexpr IINIT = -2000000000;
    static double        constexpr RINIT = -2.e12;
    static unsigned char constexpr ZINIT = 0; // false

    template <typename T>
    T getInitialValue();

public:
    DataControl();

    Data       &operator[](DataID id)       { return data[(size_type) id]; }
    Data const &operator[](DataID id) const { return data[(size_type) id]; }

    /** \brief Initialise mesh representation */
    void setMesh(Sizes const &sizes);

    /** \brief Initialise mesh quantities */
    void setQuant(Config const &config, Sizes const &sizes);

#ifdef BOOKLEAF_MPI_SUPPORT
    /** \brief Initialise inter-proc element and node IDs  */
    void setTyphon(Sizes const &sizes);
#endif

    /** \brief Resize component data stores */
    void resetCpQuant(size_type nsize, Sizes &sizes, Error &err);

    /** \brief Resize mix data stores */
    void resetMxQuant(size_type nsize, Sizes &sizes, Error &err);

#if defined BOOKLEAF_DEBUG
    /**
     * \brief Dump the contents of arrays to a compressed file amenable to
     *        diffing
     */
    void dump(std::string filename) const;
#endif

    /** \brief Sync all data to the device. */
    void syncAllDevice() const;

    /** \brief Sync all data to the host. */
    void syncAllHost() const;

private:
    std::vector<Data> data;     //!< Store data

    template <typename T>
    void
    entry(
            DataID id,
            std::string name,
            size_type num_rows,
            size_type num_cols = 1);

    template <typename T>
    void
    entry(
            DataID id,
            std::string name,
            bool zmpi,
            size_type num_rows,
            size_type num_cols = 1);

    template <typename T>
    void
    entry(
            DataID id,
            std::string name,
            bool zmpi,
            size_type mesh_dim,
            size_type num_rows,
            size_type num_cols = 1);

    template <typename T>
    void
    reset(
            DataID id,
            size_type num_rows,
            size_type num_cols = 1);
};



template <>
inline double DataControl::getInitialValue<double>() { return RINIT; }
template <>
inline int DataControl::getInitialValue<int>() { return IINIT; }
template <>
inline unsigned char DataControl::getInitialValue<unsigned char>() { return ZINIT; }



template <typename T>
void
DataControl::entry(
        DataID id,
        std::string name,
        size_type num_rows,
        size_type num_cols)
{
    (*this)[id].allocate<T>(getInitialValue<T>(), id, name, num_rows, num_cols);
}



template <typename T>
void
DataControl::entry(
        DataID id,
        std::string name,
#ifdef BOOKLEAF_MPI_SUPPORT
        bool zmpi,
#else
        bool zmpi __attribute__((unused)),
#endif
        size_type num_rows,
        size_type num_cols)
{
    static_assert(
            std::is_same<T, double>::value,
            "only double currently supported");

    Data &d = (*this)[id];
    d.allocate<T>(getInitialValue<T>(), id, name, num_rows, num_cols);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (zmpi) {
        int taddr;
        TYPH_Add_Quant(&taddr, name.c_str(), TYPH_GHOSTS_TWO,
                TYPH_DATATYPE_REAL, TYPH_CENTRING_CELL, TYPH_PURE,
                TYPH_AUXILIARY_NONE, nullptr, 0);

        int tdims[2] = { (int) num_rows, (int) num_cols };
        TYPH_Set_Quant_Address(taddr, d.data<T>(), tdims, 2);
        d.setTyphonHandle(taddr);
    }
#endif
}



template <typename T>
void
DataControl::entry(
        DataID id,
        std::string name,
#ifdef BOOKLEAF_MPI_SUPPORT
        bool zmpi,
        size_type mesh_dim,
#else
        bool zmpi __attribute__((unused)),
        size_type mesh_dim __attribute__((unused)),
#endif
        size_type num_rows,
        size_type num_cols)
{
    static_assert(
            std::is_same<T, double>::value,
            "only double currently supported");

    Data &d = (*this)[id];
    d.allocate<T>(getInitialValue<T>(), id, name, num_rows, num_cols);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (zmpi) {
        assert(0 <= mesh_dim && mesh_dim < 2);

        int tdims[2] = { (int) num_rows, (int) num_cols };

        // Tell Typhon which dimension corresponds to the mesh size
        int const prev_mesh_dim_val = tdims[mesh_dim];
        tdims[mesh_dim] = TYPH_MESH_DIM;

        int taddr;
        TYPH_Add_Quant(&taddr, name.c_str(), TYPH_GHOSTS_TWO,
                TYPH_DATATYPE_REAL, TYPH_CENTRING_CELL, TYPH_PURE,
                TYPH_AUXILIARY_NONE, tdims, 2);

        tdims[mesh_dim] = prev_mesh_dim_val;
        TYPH_Set_Quant_Address(taddr, d.data<T>(), tdims, 2);
        d.setTyphonHandle(taddr);
    }
#endif
}



template <typename T>
void
DataControl::reset(
        DataID id,
        size_type num_rows,
        size_type num_cols)
{
    (*this)[id].reallocate<T>(getInitialValue<T>(), num_rows, num_cols);
}



// Helper function for indexing 2D arrays declared as 1D arrays
template <int W>
inline int
#ifndef BOOKLEAF_DEBUG
constexpr
#endif
index2D(int j, int i) // flipped to handle the col-major indexing
{
    #ifdef BOOKLEAF_DEBUG
    if (j < 0 || j >= W) {
        std::cerr << j << " out of range (" << W << ")\n";
        assert(false);
    }
    #endif
    return i * W + j;
}

inline int
#ifndef BOOKLEAF_DEBUG
constexpr
#endif
index2D(int j, int i, int W) // flipped to handle the col-major indexing
{
    #ifdef BOOKLEAF_DEBUG
    if (j < 0 || j >= W) {
        std::cerr << j << " out of range (" << W << ")\n";
        assert(false);
    }
    #endif
    return i * W + j;
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_DATA_CONTROL_H
