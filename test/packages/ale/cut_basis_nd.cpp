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

    double const cut = 1.e-40;
    double const dencut = 1.e-6;

    int const nnd = 2626;

    ConstDeviceView<double, VarDim> ndv0(
            (double *) pre_dump[0].data, pre_dump[0].size);

    DeviceView<double, VarDim> cutv(
            (double *) pre_dump[1].data, pre_dump[1].size);
    DeviceView<double, VarDim> cutm(
            (double *) pre_dump[2].data, pre_dump[2].size);

    ale::kernel::cutBasisNd(cut, dencut, ndv0, cutv, cutm, nnd);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
