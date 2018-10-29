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
#include "packages/hydro/driver/set.h"

#include <algorithm>

#include "common/sizes.h"
#include "common/data_control.h"

#include "utilities/data/copy.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
setPredictor(
        Sizes const &sizes,
        DataControl &data)
{
    auto elenergy      = data[DataID::ELENERGY].cdevice<double, VarDim>();
    auto lag_elenergy0 = data[DataID::LAG_ELENERGY0].device<double, VarDim>();

    // Store internal energy
    utils::kernel::copy(lag_elenergy0, elenergy, sizes.nel);
}



void
setCorrector(
        Sizes const &sizes,
        DataControl &data)
{
    auto lag_elenergy0 = data[DataID::LAG_ELENERGY0].cdevice<double, VarDim>();
    auto elenergy      = data[DataID::ELENERGY].device<double, VarDim>();

    // Restore internal energy
    utils::kernel::copy(elenergy, lag_elenergy0, sizes.nel);
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
