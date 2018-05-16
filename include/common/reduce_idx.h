#ifndef BOOKLEAF_COMMON_REDUCE_IDX_H
#define BOOKLEAF_COMMON_REDUCE_IDX_H



namespace bookleaf {

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

} // namespace bookleaf



#endif // BOOKLEAF_COMMON_REDUCE_IDX_H
