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
    //    |   |4      3|
    //    |  -|---x    |
    //    |   |1  |   2|
    //   0-   x--------x
    //    |       |
    //    ----|--------|------> x
    //        0        1
    //
    // A unit-square element with a rectangle intersecting the bottom left
    // corner.

    _x[0] = 0.0;
    _x[1] = 1.0;
    _x[2] = 1.0;
    _x[3] = 0.0;

    _y[0] = 0.0;
    _y[1] = 0.0;
    _y[2] = 1.0;
    _y[3] = 1.0;

    _param[0] = -0.5;
    _param[1] = -0.5;
    _param[2] = 0.5;
    _param[3] = 0.5;

    ConstView<double, constants::NCORN> x(_x, constants::NCORN);
    ConstView<double, constants::NCORN> y(_y, constants::NCORN);

    double const vf = setup::kernel::subdivide(0, _param, x, y,
            setup::kernel::insideRectangle);

    // Volume fraction within the rectangle should equal 0.25, but the
    // approximation isn't very good.
    double const correct = 0.28770637512207031;
    bool const success = std::fabs(vf - correct) <= constants::EPSILON;

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
