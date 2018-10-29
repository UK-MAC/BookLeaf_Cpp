#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/hydro/kernel/get_energy.h"

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

    double const zerocut = 1.e-40;
    int const nel = 2500;

    ConstDeviceView<double, VarDim, NCORN> cnfx(
            (double *) pre_dump[0].data, pre_dump[0].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnfy(
            (double *) pre_dump[1].data, pre_dump[1].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnu(
            (double *) pre_dump[2].data, pre_dump[2].size / NCORN);
    ConstDeviceView<double, VarDim, NCORN> cnv(
            (double *) pre_dump[3].data, pre_dump[3].size / NCORN);

    ConstDeviceView<double, VarDim> elmass(
            (double *) pre_dump[4].data, pre_dump[4].size);

    DeviceView<double, VarDim> elenergy(
            (double *) pre_dump[5].data, pre_dump[5].size);

    hydro::kernel::getEnergy(dt, zerocut, cnfx, cnfy, cnu, cnv, elmass,
            elenergy, nel);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
