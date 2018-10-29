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
#ifndef BOOKLEAF_COMMON_DEFS_H
#define BOOKLEAF_COMMON_DEFS_H

#include <string>
#include <type_traits>

#include <RAJA/RAJA.hpp>



namespace bookleaf {

/** \brief Mark inline functions. */
#define BOOKLEAF_INLINE inline

/** \brief Consistent size type. */
typedef std::size_t SizeType;

/** \brief Mark the number of rows in a view as varying at runtime. */
SizeType constexpr VarDim = (SizeType) (-1);

namespace internal {

#ifdef BOOKLEAF_RAJA_CUDA_SUPPORT
int constexpr CUDA_BLOCK_SIZE = 1024;
#endif

/** \brief RAJA execution policy to use for host kernels. */
//using RAJA_HOST_POLICY = RAJA::omp_parallel_for_exec;

/** \brief RAJA execution policy to use for device kernels. */
#ifdef BOOKLEAF_RAJA_CUDA_SUPPORT
using RAJA_DEVICE_POLICY = RAJA::cuda_exec<CUDA_BLOCK_SIZE, false>;
#else
using RAJA_DEVICE_POLICY = RAJA::omp_parallel_for_exec;
#endif

/** \brief RAJA reduction policy for host kernels. */
//using RAJA_HOST_REDUCTION_POLICY = RAJA::omp_reduce;

/** \brief RAJA reduction policy for device kernels. */
#ifdef BOOKLEAF_RAJA_CUDA_SUPPORT
using RAJA_DEVICE_REDUCTION_POLICY = RAJA::cuda_reduce<CUDA_BLOCK_SIZE, false>;
#else
using RAJA_DEVICE_REDUCTION_POLICY = RAJA::omp_reduce;
#endif

/** \brief RAJA host layout permutation. */
using RAJA_HOST_PERMUTATION = RAJA::PERM_IJ;

/** \brief RAJA device layout permutation. */
#ifdef BOOKLEAF_RAJA_CUDA_SUPPORT
using RAJA_DEVICE_PERMUTATION = RAJA::PERM_JI;
#else
using RAJA_DEVICE_PERMUTATION = RAJA::PERM_IJ;
#endif

} // namespace internal

/** \brief Primary RAJA execution policy. */
using RAJA_POLICY = internal::RAJA_DEVICE_POLICY;

/** \brief Primary RAJA reduction policy. */
using RAJA_REDUCTION_POLICY = internal::RAJA_DEVICE_REDUCTION_POLICY;

#ifdef BOOKLEAF_RAJA_CUDA_SUPPORT
#define BOOKLEAF_DEVICE_LAMBDA [=] __device__
#define BOOKLEAF_DEVICE_FUNCTION __device__
#else
#define BOOKLEAF_DEVICE_LAMBDA [=]
#define BOOKLEAF_DEVICE_FUNCTION
#endif



/**
 * \brief   Return a string representation of the template argument type.
 */
template <typename T>
std::string
getTypeName()
{
    using nonconstT = typename std::remove_const<T>::type;

    bool constexpr is_double = std::is_same<nonconstT, double>::value;
    bool constexpr is_int    = std::is_same<nonconstT, int>::value;
    bool constexpr is_bool   = std::is_same<nonconstT, unsigned char>::value;

    static_assert(
            is_double || is_int || is_bool,
            "unsupported type");

    if      (is_double) return "double";
    else if (is_int   ) return "integer";
    else                return "boolean";
}

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_DEFS_H
