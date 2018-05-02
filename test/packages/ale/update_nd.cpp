#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/ale/kernel/advectors.h"

#include "data_dump.h"



int
main(int argc, char const *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    using constants::NCORN;
    using constants::NFACE;

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    int const nel = 2500;
    int const nnd = 2626;

    ConstView<int, VarDim, NCORN> elnd(
            (int *) pre_dump[0].data, pre_dump[0].size / NCORN);

    ConstView<double, VarDim> ndbase0(
            (double *) pre_dump[1].data, pre_dump[1].size);
    ConstView<double, VarDim> ndbase1(
            (double *) pre_dump[2].data, pre_dump[2].size);

    ConstView<double, VarDim> cut(
            (double *) pre_dump[3].data, pre_dump[3].size);

    ConstView<unsigned char, VarDim> active(
            (unsigned char *) pre_dump[4].data, pre_dump[4].size);

    ConstView<double, VarDim, NCORN> cnflux(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);

    View<double, VarDim> ndflux(
            (double *) pre_dump[6].data, pre_dump[6].size);
    View<double, VarDim> ndvar(
            (double *) pre_dump[7].data, pre_dump[7].size);

    ale::kernel::updateNd(nnd, nel, nnd, elnd, ndbase0, ndbase1, cut, active,
            cnflux, ndflux, ndvar);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
