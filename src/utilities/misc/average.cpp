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
#include "utilities/misc/average.h"

#include "common/constants.h"
#include "common/sizes.h"
#include "common/data_control.h"
#include "common/view.h"



namespace bookleaf {
namespace utils {
namespace kernel {
namespace {

using constants::NCORN;

void
average(
        int nmx,
        ConstView<int, VarDim>    imxel,
        ConstView<int, VarDim>    imxfcp,
        ConstView<int, VarDim>    imxncp,
        ConstView<double, VarDim> mxfraction,
        ConstView<double, VarDim> mxarray1,
        ConstView<double, VarDim> mxarray2,
        View<double, VarDim>      elarray)
{
    for (int imx = 0; imx < nmx; imx++) {
        double w1 = 0.;
        int const icp = imxfcp(imx);
        for (int ii = 0; ii < imxncp(imx); ii++) {
            w1 += mxfraction(icp) * (mxarray1(icp) + mxarray2(icp));
        }

        elarray(imxel(imx)) = w1;
    }
}



void
average(
        int nmx,
        ConstView<int, VarDim>           imxel,
        ConstView<int, VarDim>           imxfcp,
        ConstView<int, VarDim>           imxncp,
        ConstView<double, VarDim>        mxfraction,
        ConstView<double, VarDim, NCORN> mxarray1,
        ConstView<double, VarDim, NCORN> mxarray2,
        View<double, VarDim, NCORN>      elarray1,
        View<double, VarDim, NCORN>      elarray2)
{
    double _w1[NCORN] = {0};
    double _w2[NCORN] = {0};
    View<double, NCORN> w1(_w1);
    View<double, NCORN> w2(_w2);

    for (int imx = 0; imx < nmx; imx++) {
        int const icp = imxfcp(imx);
        for (int ii = 0; ii < imxncp(imx); ii++) {
            for (int jj = 0; jj < NCORN; jj++) {
                w1(jj) += mxfraction(icp) * mxarray1(icp, jj);
                w2(jj) += mxfraction(icp) * mxarray2(icp, jj);
            }
        }

        int const iel = imxel(imx);
        for (int icn = 0; icn < NCORN; icn++) {
            elarray1(iel, icn) = w1(icn);
            elarray2(iel, icn) = w2(icn);
        }
    }
}

} // namespace
} // namespace kernel

namespace driver {

void
average(
        Sizes const &sizes,
        DataID frid,
        DataID mx1id,
        DataID mx2id,
        DataID elid,
        DataControl &data)
{
    auto imxel  = data[DataID::IMXEL].chost<int, VarDim>();
    auto imxfcp = data[DataID::IMXFCP].chost<int, VarDim>();
    auto imxncp = data[DataID::IMXNCP].chost<int, VarDim>();
    auto fr     = data[frid].chost<double, VarDim>();
    auto mx1    = data[mx1id].chost<double, VarDim>();
    auto mx2    = data[mx2id].chost<double, VarDim>();
    auto el     = data[elid].host<double, VarDim>();

    kernel::average(sizes.nmx, imxel, imxfcp, imxncp, fr, mx1, mx2, el);
}



void
average(
        Sizes const &sizes,
        DataID frid,
        DataID mx1id,
        DataID mx2id,
        DataID el1id,
        DataID el2id,
        DataControl &data)
{
    using constants::NCORN;

    auto imxel  = data[DataID::IMXEL].chost<int, VarDim>();
    auto imxfcp = data[DataID::IMXFCP].chost<int, VarDim>();
    auto imxncp = data[DataID::IMXNCP].chost<int, VarDim>();
    auto fr     = data[frid].chost<double, VarDim>();
    auto mx1    = data[mx1id].chost<double, VarDim, NCORN>();
    auto mx2    = data[mx2id].chost<double, VarDim, NCORN>();
    auto el1    = data[el1id].host<double, VarDim, NCORN>();
    auto el2    = data[el2id].host<double, VarDim, NCORN>();

    kernel::average(sizes.nmx, imxel, imxfcp, imxncp, fr,
            mx1, mx2, el1, el2);
}

} // namespace driver
} // namespace utils
} // namespace bookleaf
