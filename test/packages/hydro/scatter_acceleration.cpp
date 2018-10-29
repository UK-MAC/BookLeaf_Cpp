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

    double const zerocut = 1.e-40;
    int const nnd = 2626;

    ConstDeviceView<int, VarDim> ndeln(
            (int *) pre_dump[0].data, pre_dump[0].size);
    ConstDeviceView<int, VarDim> ndelf(
            (int *) pre_dump[1].data, pre_dump[1].size);
    ConstDeviceView<int, VarDim> ndel(
            (int *) pre_dump[2].data, pre_dump[2].size);
    ConstDeviceView<int, VarDim, NCORN> elnd(
            (int *) pre_dump[3].data, pre_dump[3].size / NCORN);

    ConstDeviceView<double, VarDim> eldensity(
            (double *) pre_dump[4].data, pre_dump[4].size);

    ConstDeviceView<double, VarDim, NCORN> cnwt(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnmass(
            (double *) pre_dump[6].data, pre_dump[6].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnfx(
            (double *) pre_dump[7].data, pre_dump[7].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnfy(
            (double *) pre_dump[8].data, pre_dump[8].size / NCORN);

    DeviceView<double, VarDim> ndarea(
            (double *) pre_dump[9].data, pre_dump[9].size);
    DeviceView<double, VarDim> ndmass(
            (double *) pre_dump[10].data, pre_dump[10].size);
    DeviceView<double, VarDim> ndudot(
            (double *) pre_dump[11].data, pre_dump[11].size);
    DeviceView<double, VarDim> ndvdot(
            (double *) pre_dump[12].data, pre_dump[12].size);

    hydro::kernel::scatterAcceleration(zerocut, ndeln, ndelf, ndel, elnd,
            eldensity, cnwt, cnmass, cnfx, cnfy, ndarea, ndmass, ndudot, ndvdot,
            nnd);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
