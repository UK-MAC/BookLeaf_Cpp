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
#include "utilities/mix/kernel/list.h"

#include <cassert>
#include <algorithm>

#include "common/constants.h"
#include "common/data_control.h"
#include "utilities/data/sort.h"



namespace bookleaf {
namespace mix {
namespace kernel {

// Add element iel at index (nmx-1)
void
addEl(
        int iel,
        int imx,
        View<int, VarDim> elmat,
        View<int, VarDim> mxel,
        View<int, VarDim> mxncp,
        View<int, VarDim> mxfcp)
{
    mxel(imx)  = iel;           // Element ID
    mxncp(imx) = 0;             // Number of material components
    mxfcp(imx) = -1;            // Head of the component linked list
    elmat(iel) = -(imx + 1);    // Negative elmat indicates mixed element
}



// Add component to imx at index icp
void
addCp(
        int imx,
        int icp,
        View<int, VarDim> mxfcp,
        View<int, VarDim> mxncp,
        View<int, VarDim> cpprev,
        View<int, VarDim> cpnext)
{
    // Add a new component at the start of the linked list defined by cpmat,
    // cpprev and cpnext. fmix specifies the index of the original head of the
    // list. icp is the index of the new component.

    int const fmix = mxfcp(imx);

    // Store the mix index as the prev-index of the first component
    cpprev(icp) = -(imx+1);

    // Update next and prev
    cpnext(icp) = fmix;
    if (fmix != -1) cpprev(fmix) = icp;

    // Update the list head and size
    mxfcp(imx) = icp;
    mxncp(imx)++;
}



void
flattenIndex(
        int nmx,
        int ncp __attribute__((unused)),
        ConstView<int, VarDim> sort,
        ConstView<int, VarDim> mxfcp,
        ConstView<int, VarDim> mxncp,
        ConstView<int, VarDim> cpmat,
        ConstView<int, VarDim> cpnext,
        View<int, VarDim>      cpprev,
        View<int, VarDim>      index)
{
    int ii = 0;
    for (int imix = 0; imix < nmx; imix++) {
        int const jj = sort(imix);
        int icp = mxfcp(jj);
        int const lncp = mxncp(jj);
        for (int kk = 0; kk < lncp; kk++) {
            cpprev(kk) = cpmat(icp);
            cpprev(kk+lncp) = icp;
            icp = cpnext(icp);
        }

        View<int, VarDim> vicpprev1(cpprev.data, lncp);
        View<int, VarDim> vicpprev2(cpprev.data + 2*lncp, lncp);
        utils::kernel::sortIndices<int, int>(
                vicpprev1,
                vicpprev2,
                lncp);

        for (int kk = 0; kk < lncp; kk++) {
            int const ll = cpprev(2*lncp+kk);
            index(ii) = cpprev(lncp+ll);
            ii++;
        }
    }
}



void
flattenList(
        int nmx,
        int ncp __attribute__((unused)),
        ConstView<int, VarDim> sort,
        View<int, VarDim>      mxfcp,
        View<int, VarDim>      mxel,
        View<int, VarDim>      mxncp,
        View<int, VarDim>      cpprev,
        View<int, VarDim>      cpnext)
{
    // Use icpprev and imxfcp as sort scratch, then correct them at the end of
    // this routine
    for (int imx = 0; imx < nmx; imx++) {
        cpprev(imx) = mxel(imx);
        mxfcp(imx) = mxncp(imx);
    }

    // Sort imxel and imxncp and fill icpnext (linearly)
    int ii = 0;
    for (int imix = 0; imix < nmx; imix++) {
        int const jj = sort(imix);
        mxel(imix) = cpprev(jj);
        mxncp(imix) = mxfcp(jj);

        for (int kk = 0; kk < mxncp(imix); kk++) {
            cpnext(ii) = ii + 1;
            ii++;
        }
        cpnext(ii-1) = -1;
    }

    // Set imxfcp and icpprev
    ii = 0;
    for (int imix = 0; imix < nmx; imix++) {
        int jj = ii;
        for (int kk = 0; kk < mxncp(imix); kk++) {
            cpprev(ii) = ii - 1;
            ii++;
        }
        mxfcp(imix) = jj;
        cpprev(jj) = -(imix + 1);
    }
}

} // namespace kernel
} // namespace mix
} // namespace bookleaf
