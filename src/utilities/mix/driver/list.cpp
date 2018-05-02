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
#include "utilities/mix/driver/list.h"

#include <cassert>
#include <iostream>

#include "utilities/mix/kernel/list.h"
#include "utilities/data/sort.h"
#include "common/sizes.h"
#include "common/error.h"
#include "common/data_control.h"



namespace bookleaf {
namespace mix {
namespace driver {

void
resizeMx(
        Sizes &sizes,
        DataControl &data,
        int nsz,
        Error &err)
{
    if (nsz > sizes.mmx) {
        data.resetMxQuant(nsz, sizes, err);
    }
}



void
resizeCp(
        Sizes &sizes,
        DataControl &data,
        int nsz,
        Error &err)
{
    if (nsz > sizes.mcp) {
        data.resetCpQuant(nsz, sizes, err);
    }
}



void
flatten(
        Sizes const &sizes,
        DataControl &data)
{
    auto mxel       = data[DataID::IMXEL].host<int, VarDim>();
    auto mxfcp      = data[DataID::IMXFCP].host<int, VarDim>();
    auto mxncp      = data[DataID::IMXNCP].host<int, VarDim>();
    auto cpmat      = data[DataID::ICPMAT].host<int, VarDim>();
    auto cpnext     = data[DataID::ICPNEXT].host<int, VarDim>();
    auto cpprev     = data[DataID::ICPPREV].host<int, VarDim>();
    auto scratch    = data[DataID::ISCRATCH11].host<int, VarDim>();
    auto cpscratch1 = data[DataID::ICPSCRATCH11].host<int, VarDim>();
    auto cpscratch2 = data[DataID::ICPSCRATCH12].host<int, VarDim>();
    auto rcpscratch = data[DataID::RCPSCRATCH11].host<double, VarDim>();
    auto frvolume   = data[DataID::FRVOLUME].host<double, VarDim>();

    // Sort mixed elements by cell list
    utils::kernel::sortIndices<int>(mxel, scratch, sizes.nmx);

    // Set new connectivity
    kernel::flattenIndex(sizes.nmx, sizes.ncp, scratch, mxfcp, mxncp, cpmat,
            cpnext, cpprev, cpscratch1);

    kernel::flattenList(sizes.nmx, sizes.ncp, scratch, mxfcp, mxel, mxncp,
            cpprev, cpnext);

    kernel::flattenQuant(sizes.ncp, cpscratch1, cpscratch2, cpmat);

    kernel::flattenQuant(sizes.ncp, cpscratch1, rcpscratch, frvolume);
}

} // namespace driver
} // namespace mix
} // namespace bookleaf
