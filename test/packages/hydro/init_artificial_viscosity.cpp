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

    ConstDeviceView<double, VarDim, NCORN> cnx(
            (double *) pre_dump[0].data, pre_dump[0].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cny(
            (double *) pre_dump[1].data, pre_dump[1].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnu(
            (double *) pre_dump[2].data, pre_dump[2].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnv(
            (double *) pre_dump[3].data, pre_dump[3].size / NCORN);

    DeviceView<double, VarDim> elvisc(
            (double *) pre_dump[4].data, pre_dump[4].size);

    DeviceView<double, VarDim, NFACE> dx(
            (double *) pre_dump[5].data, pre_dump[5].size / NFACE);
    DeviceView<double, VarDim, NFACE> dy(
            (double *) pre_dump[6].data, pre_dump[6].size / NFACE);
    DeviceView<double, VarDim, NFACE> du(
            (double *) pre_dump[7].data, pre_dump[7].size / NFACE);
    DeviceView<double, VarDim, NFACE> dv(
            (double *) pre_dump[8].data, pre_dump[8].size / NFACE);

    DeviceView<double, VarDim, NFACE> cnviscx(
            (double *) pre_dump[9].data, pre_dump[9].size / NFACE);
    DeviceView<double, VarDim, NFACE> cnviscy(
            (double *) pre_dump[10].data, pre_dump[10].size / NFACE);

    hydro::kernel::initArtificialViscosity(cnx, cny, cnu, cnv, elvisc, dx, dy,
            du, dv, cnviscx, cnviscy, nel);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
