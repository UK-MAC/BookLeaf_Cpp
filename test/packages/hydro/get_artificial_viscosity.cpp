#include <iostream>

#include "common/constants.h"
#include "common/view.h"
#include "packages/hydro/kernel/get_artificial_viscosity.h"

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

    double const zerocut = 1.e-40;
    int const nel = 2500;

    ConstView<double, VarDim, NCORN> cnx(
            (double *) pre_dump[0].data, pre_dump[0].size / NCORN, NCORN);
    ConstView<double, VarDim, NCORN> cny(
            (double *) pre_dump[1].data, pre_dump[1].size / NCORN, NCORN);
    ConstView<double, VarDim, NCORN> cnu(
            (double *) pre_dump[2].data, pre_dump[2].size / NCORN, NCORN);
    ConstView<double, VarDim, NCORN> cnv(
            (double *) pre_dump[3].data, pre_dump[3].size / NCORN, NCORN);

    View<double, VarDim, NFACE> cnviscx(
            (double *) pre_dump[4].data, pre_dump[4].size / NFACE, NFACE);
    View<double, VarDim, NFACE> cnviscy(
            (double *) pre_dump[5].data, pre_dump[5].size / NFACE, NFACE);

    hydro::kernel::getArtificialViscosity(zerocut, cnx, cny, cnu, cnv, cnviscx,
            cnviscy, nel);

    bool const success = pre_dump.diff(post_dump);
    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
