#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/hydro/kernel/get_force.h"

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

    int const nel = 2500;

    ConstDeviceView<double, VarDim> elpressure(
            (double *) pre_dump[0].data, pre_dump[0].size);
    ConstDeviceView<double, VarDim> a1(
            (double *) pre_dump[1].data, pre_dump[1].size);
    ConstDeviceView<double, VarDim> a3(
            (double *) pre_dump[2].data, pre_dump[2].size);
    ConstDeviceView<double, VarDim> b1(
            (double *) pre_dump[3].data, pre_dump[3].size);
    ConstDeviceView<double, VarDim> b3(
            (double *) pre_dump[4].data, pre_dump[4].size);

    DeviceView<double, VarDim, NCORN> cnfx(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);
    DeviceView<double, VarDim, NCORN> cnfy(
            (double *) pre_dump[6].data, pre_dump[6].size / NCORN);

    hydro::kernel::getForcePressure(elpressure, a1, a3, b1, b3, cnfx, cnfy,
            nel);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
