#include <cmath>

#include "common/constants.h"
#include "common/view.h"
#include "packages/setup/kernel/shapes.h"



int
main(
        int argc __attribute__((unused)),
        char const *argv[] __attribute__((unused)))
{
    using namespace bookleaf;

    double _x[constants::NCORN];
    double _y[constants::NCORN];

    double _param[4];

    //    y
    //    ^
    //   1-   x--------x
    //    |   |4  \   3|
    //    |   |    x   |
    //    |   |1  /   2|
    //   0-   x--------x
    //    |
    //    ----|--------|------> x
    //        0        1
    //
    // A unit-square element with a circle intersecting the left half.

    _x[0] = 0.0;
    _x[1] = 1.0;
    _x[2] = 1.0;
    _x[3] = 0.0;

    _y[0] = 0.0;
    _y[1] = 0.0;
    _y[2] = 1.0;
    _y[3] = 1.0;

    _param[0] = 0.0;
    _param[1] = 0.5;
    _param[2] = 0.6;

    ConstView<double, constants::NCORN> x(_x);
    ConstView<double, constants::NCORN> y(_y);

    int const nnd = setup::kernel::intersect(_param, x, y,
            setup::kernel::insideCircle);

    // The left two nodes are inside the circle.
    if (nnd == 2) return EXIT_SUCCESS;
    return EXIT_FAILURE;
}
