#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/ale/kernel/advect.h"

#include "data_dump.h"



int
main(int argc, char const *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    using constants::NCORN;

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    int const nnd = 2626;

    ConstView<int, VarDim, NCORN> elnd(
            (int *) pre_dump[0].data, pre_dump[0].size / NCORN, NCORN);
    ConstView<int, VarDim> ndeln(
            (int *) pre_dump[1].data, pre_dump[1].size);
    ConstView<int, VarDim> ndelf(
            (int *) pre_dump[2].data, pre_dump[2].size);
    ConstView<int, VarDim> ndel(
            (int *) pre_dump[3].data, pre_dump[3].size);

    ConstView<double, VarDim> elv0(
            (double *) pre_dump[4].data, pre_dump[4].size);
    ConstView<double, VarDim> elv1(
            (double *) pre_dump[5].data, pre_dump[5].size);

    ConstView<double, VarDim, NCORN> cnm1(
            (double *) pre_dump[6].data, pre_dump[6].size / NCORN, NCORN);

    View<double, VarDim> ndv0(
            (double *) pre_dump[7].data, pre_dump[7].size);
    View<double, VarDim> ndv1(
            (double *) pre_dump[8].data, pre_dump[8].size);
    View<double, VarDim> ndm0(
            (double *) pre_dump[9].data, pre_dump[9].size);

    ale::kernel::calcBasisNd(elnd, ndeln, ndelf, ndel, elv0, elv1, cnm1, ndv0,
            ndv1, ndm0, nnd);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
