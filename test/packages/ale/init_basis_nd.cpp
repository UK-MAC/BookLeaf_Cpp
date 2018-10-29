#include <iostream>

#include "common/view.h"
#include "packages/ale/kernel/advect.h"

#include "data_dump.h"



int
main(int argc, char *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    Kokkos::initialize(argc, argv);

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    int const nnd = 2626;

    DeviceView<double, VarDim> ndv0(
            (double *) pre_dump[0].data, pre_dump[0].size);
    DeviceView<double, VarDim> ndv1(
            (double *) pre_dump[1].data, pre_dump[1].size);
    DeviceView<double, VarDim> ndm0(
            (double *) pre_dump[2].data, pre_dump[2].size);

    ale::kernel::initBasisNd(ndv0, ndv1, ndm0, nnd);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
