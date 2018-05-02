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
#include "packages/ale/driver/get_dt.h"

#include "packages/ale/kernel/get_dt.h"
#include "packages/ale/config.h"
#include "common/sizes.h"
#include "common/dt.h"
#include "utilities/data/global_configuration.h"
#include "common/data_control.h"



namespace bookleaf {
namespace ale {
namespace driver {

void
getDt(ale::Config const &ale, Sizes const &sizes, DataControl &data, Dt *&dt)
{
    using constants::NCORN;

    auto cnu   = data[DataID::TIME_CNU].chost<double, VarDim, NCORN>();
    auto cnv   = data[DataID::TIME_CNV].chost<double, VarDim, NCORN>();
    auto ellen = data[DataID::TIME_ELLENGTH].chost<double, VarDim>();

    dt->next = new Dt();
    dt = dt->next;

    // Calculate ALE timestep control
    kernel::getDt(sizes.nel, ale.global->zerocut, ale.sf, ale.zeul, cnu, cnv,
            ellen, dt->rdt, dt->idt, dt->sdt);
}

} // namespace driver
} // namespace ale
} // namespace bookleaf
