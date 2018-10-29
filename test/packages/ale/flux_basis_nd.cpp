#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/ale/kernel/advect.h"

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

    int const id1 = 1;
    int const id2 = 2;
    int const nel = 2500;

    ConstDeviceView<int, VarDim, NFACE> elel(
            (int *) pre_dump[0].data, pre_dump[0].size / NFACE);
    ConstDeviceView<int, VarDim, NFACE> elfc(
            (int *) pre_dump[1].data, pre_dump[1].size / NFACE);
    ConstDeviceView<int, VarDim> elsort(
            (int *) pre_dump[2].data, pre_dump[2].size);

    ConstDeviceView<double, VarDim, NFACE> fcdv(
            (double *) pre_dump[3].data, pre_dump[3].size / NFACE);
    ConstDeviceView<double, VarDim, NFACE> fcdm(
            (double *) pre_dump[4].data, pre_dump[4].size / NFACE);

    DeviceView<double, VarDim, NCORN> cndv(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);
    DeviceView<double, VarDim, NCORN> cndm(
            (double *) pre_dump[6].data, pre_dump[6].size / NCORN);
    DeviceView<double, VarDim, NCORN> cnflux(
            (double *) pre_dump[7].data, pre_dump[7].size / NCORN);

    ale::kernel::fluxBasisNd(id1, id2, elel, elfc, elsort, fcdv, fcdm, cndv,
            cndm, cnflux, nel);

    Kokkos::finalize();

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
