#include <iostream>

#include "common/view.h"
#include "packages/ale/kernel/advect.h"

#include "data_dump.h"



int
main(int argc, char const *argv[])
{
    using namespace bookleaf;
    using namespace bookleaf_diff;

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

    ConstView<double, VarDim> totv(
            (double *) pre_dump[0].data, pre_dump[0].size);
    ConstView<double, VarDim> totm(
            (double *) pre_dump[1].data, pre_dump[1].size);

    View<double, VarDim> cutv(
            (double *) pre_dump[2].data, pre_dump[2].size);
    View<double, VarDim> cutm(
            (double *) pre_dump[3].data, pre_dump[3].size);
    View<double, VarDim> elvpr(
            (double *) pre_dump[4].data, pre_dump[4].size);
    View<double, VarDim> elmpr(
            (double *) pre_dump[5].data, pre_dump[5].size);
    View<double, VarDim> eldpr(
            (double *) pre_dump[6].data, pre_dump[6].size);
    View<double, VarDim> elvolume(
            (double *) pre_dump[7].data, pre_dump[7].size);
    View<double, VarDim> elmass(
            (double *) pre_dump[8].data, pre_dump[8].size);
    View<double, VarDim> eldensity(
            (double *) pre_dump[9].data, pre_dump[9].size);

    ale::kernel::updateBasisEl(zerocut, dencut, totv, totm, cutv, cutm, elvpr,
            elmpr, eldpr, elvolume, elmass, eldensity, nel);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
