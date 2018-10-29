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

    double const dencut = 1.e-6;
    double const zerocut = 1.e-40;

    int const nel = 2500;

    ConstDeviceView<double, VarDim> totv(
            (double *) pre_dump[0].data, pre_dump[0].size);
    ConstDeviceView<double, VarDim> totm(
            (double *) pre_dump[1].data, pre_dump[1].size);

    DeviceView<double, VarDim> cutv(
            (double *) pre_dump[2].data, pre_dump[2].size);
    DeviceView<double, VarDim> cutm(
            (double *) pre_dump[3].data, pre_dump[3].size);
    DeviceView<double, VarDim> elvpr(
            (double *) pre_dump[4].data, pre_dump[4].size);
    DeviceView<double, VarDim> elmpr(
            (double *) pre_dump[5].data, pre_dump[5].size);
    DeviceView<double, VarDim> eldpr(
            (double *) pre_dump[6].data, pre_dump[6].size);
    DeviceView<double, VarDim> elvolume(
            (double *) pre_dump[7].data, pre_dump[7].size);
    DeviceView<double, VarDim> elmass(
            (double *) pre_dump[8].data, pre_dump[8].size);
    DeviceView<double, VarDim> eldensity(
            (double *) pre_dump[9].data, pre_dump[9].size);

    ale::kernel::updateBasisEl(zerocut, dencut, totv, totm, cutv, cutm, elvpr,
            elmpr, eldpr, elvolume, elmass, eldensity, nel);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
