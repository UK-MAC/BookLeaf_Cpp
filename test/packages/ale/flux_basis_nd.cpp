#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/ale/kernel/advect.h"

#include "data_dump.h"



int
main(int argc, char const *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

    using constants::NCORN;
    using constants::NFACE;

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

    ConstView<int, VarDim, NFACE> elel(
            (int *) pre_dump[0].data, pre_dump[0].size / NFACE);
    ConstView<int, VarDim, NFACE> elfc(
            (int *) pre_dump[1].data, pre_dump[1].size / NFACE);
    ConstView<int, VarDim> elsort(
            (int *) pre_dump[2].data, pre_dump[2].size);

    ConstView<double, VarDim, NFACE> fcdv(
            (double *) pre_dump[3].data, pre_dump[3].size / NFACE);
    ConstView<double, VarDim, NFACE> fcdm(
            (double *) pre_dump[4].data, pre_dump[4].size / NFACE);

    View<double, VarDim, NCORN> cndv(
            (double *) pre_dump[5].data, pre_dump[5].size / NCORN);
    View<double, VarDim, NCORN> cndm(
            (double *) pre_dump[6].data, pre_dump[6].size / NCORN);
    View<double, VarDim, NCORN> cnflux(
            (double *) pre_dump[7].data, pre_dump[7].size / NCORN);

    ale::kernel::fluxBasisNd(id1, id2, elel, elfc, elsort, fcdv, fcdm, cndv,
            cndm, cnflux, nel);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
