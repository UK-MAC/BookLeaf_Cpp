#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/hydro/kernel/get_artificial_viscosity.h"

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
    double const zerocut = 1.e-40;
    double const cvisc1 = 0.5;
    double const cvisc2 = 0.75;

    ConstDeviceView<int, VarDim> ndtype(
            (int *) pre_dump[0].data, pre_dump[0].size);

    ConstDeviceView<int, VarDim, NFACE> elel(
            (int *) pre_dump[1].data, pre_dump[1].size / NFACE);
    ConstDeviceView<int, VarDim, NCORN> elnd(
            (int *) pre_dump[2].data, pre_dump[2].size / NCORN);
    ConstDeviceView<int, VarDim, NFACE> elfc(
            (int *) pre_dump[3].data, pre_dump[3].size / NFACE);

    ConstDeviceView<double, VarDim> eldensity(
            (double *) pre_dump[4].data, pre_dump[4].size);
    ConstDeviceView<double, VarDim> elcs2(
            (double *) pre_dump[5].data, pre_dump[5].size);

    ConstDeviceView<double, VarDim, NFACE> du(
            (double *) pre_dump[6].data, pre_dump[6].size / NFACE);
    ConstDeviceView<double, VarDim, NFACE> dv(
            (double *) pre_dump[7].data, pre_dump[7].size / NFACE);
    ConstDeviceView<double, VarDim, NFACE> dx(
            (double *) pre_dump[8].data, pre_dump[8].size / NFACE);
    ConstDeviceView<double, VarDim, NFACE> dy(
            (double *) pre_dump[9].data, pre_dump[9].size / NFACE);

    DeviceView<double, VarDim, NCORN> scratch(
            (double *) pre_dump[10].data, pre_dump[10].size / NCORN);

    DeviceView<double, VarDim, NFACE> cnviscx(
            (double *) pre_dump[11].data, pre_dump[11].size / NFACE);
    DeviceView<double, VarDim, NFACE> cnviscy(
            (double *) pre_dump[12].data, pre_dump[12].size / NFACE);

    DeviceView<double, VarDim> elvisc(
            (double *) pre_dump[13].data, pre_dump[13].size);

    hydro::kernel::limitArtificialViscosity(nel, zerocut, cvisc1, cvisc2,
            ndtype, elel, elnd, elfc, eldensity, elcs2, du, dv, dx, dy, scratch,
            cnviscx, cnviscy, elvisc);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
