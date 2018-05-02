#include <iostream>

#include "common/view.h"
#include "packages/ale/kernel/advect.h"

#include "data_dump.h"



int
main(int argc, char const *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    int const ibc = -2;
    int const nnd = 2626;

    ConstView<int, VarDim> ndstatus(
            (int *) pre_dump[0].data, pre_dump[0].size);
    ConstView<int, VarDim> ndtype(
            (int *) pre_dump[1].data, pre_dump[1].size);

    View<unsigned char, VarDim> active(
            (unsigned char *) pre_dump[2].data, pre_dump[2].size);

    ale::kernel::activeNd(ibc, ndstatus, ndtype, active, nnd);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
