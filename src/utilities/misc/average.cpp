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
        ConstDeviceView<int, VarDim>    imxel,
        ConstDeviceView<int, VarDim>    imxfcp,
        ConstDeviceView<int, VarDim>    imxncp,
        ConstDeviceView<double, VarDim> mxfraction,
        ConstDeviceView<double, VarDim> mxarray1,
        ConstDeviceView<double, VarDim> mxarray2,
        DeviceView<double, VarDim>      elarray)
{
    Kokkos::parallel_for(
            RangePolicy(0, nmx),
            KOKKOS_LAMBDA (int const imx)
    {
        double w1 = 0.;
        int icp = imxfcp(imx);
        for (int ii = 0; ii < imxncp(imx); ii++) {
            w1 += mxfraction(icp) * (mxarray1(icp) + mxarray2(icp));
            icp++;
        }

        elarray(imxel(imx)) = w1;
    });
}



void
average(
        int nmx,
        ConstDeviceView<int, VarDim>           imxel,
        ConstDeviceView<int, VarDim>           imxfcp,
        ConstDeviceView<int, VarDim>           imxncp,
        ConstDeviceView<double, VarDim>        mxfraction,
        ConstDeviceView<double, VarDim, NCORN> mxarray1,
        ConstDeviceView<double, VarDim, NCORN> mxarray2,
        DeviceView<double, VarDim, NCORN>      elarray1,
        DeviceView<double, VarDim, NCORN>      elarray2)
{
    Kokkos::parallel_for(
            RangePolicy(0, nmx),
            KOKKOS_LAMBDA (int const imx)
    {
        double w1[NCORN] = {0};
        double w2[NCORN] = {0};

        int icp = imxfcp(imx);
        for (int ii = 0; ii < imxncp(imx); ii++) {
            for (int icn = 0; icn < NCORN; icn++) {
                w1[icn] += mxfraction(icp) * mxarray1(icp, icn);
                w2[icn] += mxfraction(icp) * mxarray2(icp, icn);
            }
            icp++;
        }

        int const iel = imxel(imx);
        for (int icn = 0; icn < NCORN; icn++) {
            elarray1(iel, icn) = w1[icn];
            elarray2(iel, icn) = w2[icn];
        }
    });
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
    auto imxel  = data[DataID::IMXEL].cdevice<int, VarDim>();
    auto imxfcp = data[DataID::IMXFCP].cdevice<int, VarDim>();
    auto imxncp = data[DataID::IMXNCP].cdevice<int, VarDim>();
    auto fr     = data[frid].cdevice<double, VarDim>();
    auto mx1    = data[mx1id].cdevice<double, VarDim>();
    auto mx2    = data[mx2id].cdevice<double, VarDim>();
    auto el     = data[elid].device<double, VarDim>();

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

    auto imxel  = data[DataID::IMXEL].cdevice<int, VarDim>();
    auto imxfcp = data[DataID::IMXFCP].cdevice<int, VarDim>();
    auto imxncp = data[DataID::IMXNCP].cdevice<int, VarDim>();
    auto fr     = data[frid].cdevice<double, VarDim>();
    auto mx1    = data[mx1id].cdevice<double, VarDim, NCORN>();
    auto mx2    = data[mx2id].cdevice<double, VarDim, NCORN>();
    auto el1    = data[el1id].device<double, VarDim, NCORN>();
    auto el2    = data[el2id].device<double, VarDim, NCORN>();

    kernel::average(sizes.nmx, imxel, imxfcp, imxncp, fr,
            mx1, mx2, el1, el2);
}

} // namespace driver
} // namespace utils
} // namespace bookleaf
