#include <cmath>

#include "common/constants.h"
#include "common/view.h"
#include "utilities/geometry/geometry.h"



int
main(
        int argc __attribute__((unused)),
        char const *argv[] __attribute__((unused)))
{
    using namespace bookleaf;

    double _x[constants::NCORN];
    double _y[constants::NCORN];

    //    y
    //    ^
    //   1-   x--------x
    //    |   |4      3|
    //    |   |        |
    //    |   |1      2|
    //   0-   x--------x
    //    |
    //    ----|--------|------> x
    //        0        1

    _x[0] = 0.0;
    _x[1] = 1.0;
    _x[2] = 1.0;
    _x[3] = 0.0;

    _y[0] = 0.0;
    _y[1] = 0.0;
    _y[2] = 1.0;
    _y[3] = 1.0;

    ConstView<double, constants::NCORN> x(_x, constants::NCORN);
    ConstView<double, constants::NCORN> y(_y, constants::NCORN);

    double _centroid[constants::NDIM];
    View<double, constants::NDIM> centroid(_centroid, constants::NDIM);

    geometry::kernel::getCentroid(x, y, centroid);

    // Centroid should be at (0.5, 0.5)
    bool const success =
        (std::fabs(centroid(0) - 0.5) <= constants::EPSILON) &&
        (std::fabs(centroid(1) - 0.5) <= constants::EPSILON);

    return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
