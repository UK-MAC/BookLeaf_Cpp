#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/hydro/kernel/get_acceleration.h"

#include "data_dump.h"



int
main(int argc, char *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    using constants::NCORN;
    using constants::NFACE;

    Kokkos::initialize(argc, argv);

    if (argc != 3) {
        std::cerr << "incorrect args\n";
        return EXIT_FAILURE;
    }

    DataDump pre_dump;
    DataDump post_dump;
    pre_dump.read(argv[1]);
    post_dump.read(argv[2]);

    // Get exact double from hex representation for dt value.
    double dt;
    {
        uint64_t idt = 0x3fb999999999999a;
        double *pdt  = (double *) &idt;
        dt = *pdt;
    }

    int const nnd = 2626;

    DeviceView<double, VarDim> ndubar(
            (double *) pre_dump[0].data, pre_dump[0].size);
    DeviceView<double, VarDim> ndvbar(
            (double *) pre_dump[1].data, pre_dump[1].size);
    DeviceView<double, VarDim> ndu(
            (double *) pre_dump[2].data, pre_dump[2].size);
    DeviceView<double, VarDim> ndv(
            (double *) pre_dump[3].data, pre_dump[3].size);

    hydro::kernel::applyAcceleration(dt, ndubar, ndvbar, ndu, ndv, nnd);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
