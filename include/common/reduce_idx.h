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
#ifndef BOOKLEAF_COMMON_REDUCE_IDX_H
#define BOOKLEAF_COMMON_REDUCE_IDX_H



namespace bookleaf {

#ifdef BOOKLEAF_OPENMP_USER_REDUCTIONS

/** \brief Helper for performing reductions over a value and associated index. */
struct ReduceIdx
{
    double val;
    int idx;

    bool operator<(ReduceIdx const &rhs) const { return val < rhs.val; }
    bool operator>(ReduceIdx const &rhs) const { return val > rhs.val; }
};

// Declare OpenMP reductions over this type
#pragma omp declare reduction \
    (minloc : ReduceIdx : omp_out = omp_in.val < omp_out.val ? omp_in : omp_out) \
    initializer(omp_priv=ReduceIdx { std::numeric_limits<double>::max(), -1 })

#pragma omp declare reduction \
    (maxloc : ReduceIdx : omp_out = omp_in.val > omp_out.val ? omp_in : omp_out) \
    initializer(omp_priv=ReduceIdx { -std::numeric_limits<double>::max(), -1 })

#endif // BOOKLEAF_OPENMP_USER_REDUCTIONS

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_REDUCE_IDX_H
