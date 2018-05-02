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

    using constants::NFACE;

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    int const id1 = 1;
    int const id2 = 2;
    int const nel = 2500;

    ConstView<int, VarDim, NFACE> elel(
            (int *) pre_dump[0].data, pre_dump[0].size / NFACE);
    ConstView<int, VarDim, NFACE> elfc(
            (int *) pre_dump[1].data, pre_dump[1].size / NFACE);

    ConstView<double, VarDim, NFACE> fcflux(
            (double *) pre_dump[2].data, pre_dump[2].size / NFACE);

    View<double, VarDim> elflux(
            (double *) pre_dump[3].data, pre_dump[3].size);

    ale::kernel::sumFlux(id1, id2, nel, nel, elel, elfc, fcflux, elflux);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
