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
#ifndef BOOKLEAF_UTILITIES_DATA_GATHER_H
#define BOOKLEAF_UTILITIES_DATA_GATHER_H

#ifdef BOOKLEAF_CALIPER_SUPPORT
#include <caliper/cali.h>
#endif

#include "common/constants.h"
#include "common/sizes.h"
#include "common/data_id.h"
#include "common/data_control.h"
#include "common/view.h"



namespace bookleaf {
namespace utils {
namespace kernel {

using constants::NCORN;

template <typename T>
void
cornerGather(
        int nel,
        ConstView<int, VarDim, NCORN> elnd,
        ConstView<T, VarDim>          nd,
        View<T, VarDim, NCORN>        el)
{
#ifdef BOOKLEAF_CALIPER_SUPPORT
    CALI_CXX_MARK_FUNCTION;
#endif

    #pragma omp parallel for
    for (int i = 0; i < nel; i++) {
        for (int j = 0; j < NCORN; j++) {
            el(i, j) = nd(elnd(i, j));
        }
    }
}



template <typename T>
void
mxGather(
        int nmx,
        ConstView<int, VarDim> imxel,
        ConstView<int, VarDim> imxfcp,
        ConstView<int, VarDim> imxncp,
        ConstView<T, VarDim>   elarray,
        View<T, VarDim>        mxarray)
{
    #pragma omp parallel for
    for (int imx = 0; imx < nmx; imx++) {
        T const w = elarray(imxel(imx));
        for (int icp = imxfcp(imx); icp < imxfcp(imx) + imxncp(imx); icp++) {
            mxarray(icp) = w;
        }
    }
}



template <typename T>
void
mxGather(
        int nmx,
        ConstView<int, VarDim> imxel,
        ConstView<int, VarDim> imxfcp,
        ConstView<int, VarDim> imxncp,
        ConstView<T, VarDim>   elarray1,
        ConstView<T, VarDim>   elarray2,
        ConstView<T, VarDim>   elarray3,
        ConstView<T, VarDim>   elarray4,
        View<T, VarDim>        mxarray1,
        View<T, VarDim>        mxarray2,
        View<T, VarDim>        mxarray3,
        View<T, VarDim>        mxarray4)
{
    #pragma omp parallel for
    for (int imx = 0; imx < nmx; imx++) {
        int const iel = imxel(imx);
        T const w1 = elarray1(iel);
        T const w2 = elarray2(iel);
        T const w3 = elarray3(iel);
        T const w4 = elarray4(iel);
        for (int icp = imxfcp(imx); icp < imxfcp(imx) + imxncp(imx); icp++) {
            mxarray1(icp) = w1;
            mxarray2(icp) = w2;
            mxarray3(icp) = w3;
            mxarray4(icp) = w4;
        }
    }
}



template <typename T>
void
mxCornerGather(
        int nmx,
        ConstView<int, VarDim>      imxel,
        ConstView<int, VarDim>      imxfcp,
        ConstView<int, VarDim>      imxncp,
        ConstView<T, VarDim, NCORN> elarray,
        View<T, VarDim, NCORN>      mxarray)
{
    #pragma omp parallel for
    for (int imx = 0; imx < nmx; imx++) {
        T w[NCORN];

        int const iel = imxel(imx);
        for (int j = 0; j < NCORN; j++) {
            w[j] = elarray(iel, j);
        }

        for (int icp = imxfcp(imx); icp < imxfcp(imx) + imxncp(imx); icp++) {
            for (int j = 0; j < NCORN; j++) {
                mxarray(icp, j) = w[j];
            }
        }
    }
}

} // namespace kernel

namespace driver {

template <typename T = double>
void
cornerGather(
        Sizes const &sizes,
        DataID ndid,
        DataID elid,
        DataControl &data)
{
    using constants::NCORN;

    auto elnd = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto nd   = data[ndid].chost<T, VarDim>();
    auto el   = data[elid].host<T, VarDim, NCORN>();

    kernel::cornerGather<T>(
            sizes.nel,
            elnd,
            nd,
            el);
}

} // namespace driver
} // namespace utils
} // namespace bookleaf



#endif // BOOKLEAF_UTILITIES_DATA_GATHER_H
