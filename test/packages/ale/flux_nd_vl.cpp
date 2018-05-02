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

    ConstView<int, VarDim, NFACE> elel(
            (int *) pre_dump[0].data, pre_dump[0].size / NFACE);
    ConstView<int, VarDim, NFACE> elfc(
            (int *) pre_dump[1].data, pre_dump[1].size / NFACE);

    ConstView<double, VarDim, NCORN> cnbasis(
            (double *) pre_dump[2].data, pre_dump[2].size / NCORN);
    ConstView<double, VarDim, NCORN> cndbasis(
            (double *) pre_dump[3].data, pre_dump[3].size / NCORN);

    ConstView<double, VarDim, NCORN> cnvar(
            (double *) pre_dump[4].data, pre_dump[4].size / NCORN);

    View<double, VarDim, NCORN> cnflux(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);

    ale::kernel::fluxNdVl(nel, nel, elel, elfc, cnbasis, cndbasis, cnvar,
            cnflux);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
