/* @HEADER@
 * Crown Copyright 2018 AWE.
 *
 * This file is part of BookLeaf.
 *
 * BookLeaf is free software: you can redistribute it and/or modify it under the
 * terms of the GNU General Public License as published by the Free Software
 * Foundation, either version 3 of the License, or (at your option) any later
 * version.
 * 
 * BookLeaf is distributed in the hope that it will be useful, but WITHOUT ANY
 * WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR
 * A PARTICULAR PURPOSE. See the GNU General Public License for more details.
 * 
 * You should have received a copy of the GNU General Public License along with
 * BookLeaf. If not, see http://www.gnu.org/licenses/.
 * @HEADER@ */
#include "utilities/misc/boundary_conditions.h"

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "utilities/data/global_configuration.h"
#include "common/data_control.h"
#include "common/view.h"
#include "common/sizes.h"



namespace bookleaf {
namespace utils {
namespace kernel {
namespace {

void
setBoundaryConditions(
        int nnd,
        double rcut,
        ConstView<int, VarDim> ndtype,
        View<double, VarDim>   ndu,
        View<double, VarDim>   ndv)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    double const w1 = rcut*rcut;
    RAJA::forall<RAJA_POLICY>(
            RAJA::RangeSegment(0, nnd),
            BOOKLEAF_DEVICE_LAMBDA (int const ind)
    {
        switch (ndtype(ind)) {
        case -1:
            ndu(ind) = 0.0;
            break;
        case -2:
            ndv(ind) = 0.0;
            break;
        case -3:
            ndu(ind) = 0.0;
            ndv(ind) = 0.0;
            break;
        default:
            break;
        }

        double const w2 = ndu(ind)*ndu(ind) + ndv(ind)*ndv(ind);
        ndu(ind) = w2 < w1 ? 0.0 : ndu(ind);
        ndv(ind) = w2 < w1 ? 0.0 : ndv(ind);
    });
}

} // namespace
} // namespace kernel

namespace driver {

void
setBoundaryConditions(
        GlobalConfiguration const &global,
        Sizes const &sizes,
        DataID idx,
        DataID idy,
        DataControl &data)
{
    auto ndtype = data[DataID::INDTYPE].cdevice<int, VarDim>();
    auto x      = data[idx].device<double, VarDim>();
    auto y      = data[idy].device<double, VarDim>();

    kernel::setBoundaryConditions(sizes.nnd, global.accut, ndtype, x, y);
}

} // namespace driver
} // namespace utils
} // namespace bookleaf
