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
#include "packages/hydro/driver/get_dt.h"

#include "common/sizes.h"
#include "common/dt.h"
#include "packages/hydro/kernel/get_dt.h"
#include "packages/hydro/driver/get.h"
#include "packages/hydro/config.h"
#include "packages/hydro/driver/get_cs2.h"
#include "utilities/data/gather.h"
#include "utilities/data/global_configuration.h"
#include "common/data_control.h"



namespace bookleaf {
namespace hydro {
namespace driver {

void
getDt(
        hydro::Config const &hydro,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data,
        Dt *&dt,
        Error &err)
{
    using constants::NCORN;

    auto elreg    = data[DataID::IELREG].chost<int, VarDim>();
    auto elcs2    = data[DataID::ELCS2].chost<double, VarDim>();
    auto cnx      = data[DataID::CNX].chost<double, VarDim, NCORN>();
    auto cny      = data[DataID::CNY].chost<double, VarDim, NCORN>();
    auto scratch  = data[DataID::TIME_SCRATCH].host<double, VarDim>();
    auto ellen    = data[DataID::TIME_ELLENGTH].host<double, VarDim>();
    auto a1       = data[DataID::A1].chost<double, VarDim>();
    auto a3       = data[DataID::A3].chost<double, VarDim>();
    auto b1       = data[DataID::B1].chost<double, VarDim>();
    auto b3       = data[DataID::B3].chost<double, VarDim>();
    auto elvolume = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto cnu      = data[DataID::TIME_CNU].chost<double, VarDim, NCORN>();
    auto cnv      = data[DataID::TIME_CNV].chost<double, VarDim, NCORN>();

    utils::driver::cornerGather(sizes, DataID::NDU, DataID::TIME_CNU, data);
    utils::driver::cornerGather(sizes, DataID::NDV, DataID::TIME_CNV, data);

    // FIXME(timrlaw): It's a bit unexpected for this call to be here, maybe
    //                 consider moving it to the top-level in lagstep if
    //                 possible.
    getArtificialViscosity(hydro, sizes, timers, data);

    getCs2(sizes, data);

    // Courant–Friedrichs–Lewy condition
    dt->next = new Dt();
    dt = dt->next;
    hydro::kernel::getDtCfl(sizes.nel, hydro.global->zcut, hydro.cfl_sf,
            hydro.zdtnotreg.data(), hydro.zmidlength.data(), elreg, elcs2, cnx,
            cny, scratch, ellen, dt->rdt, dt->idt, dt->sdt, err);

    if (err.failed()) {
        FAIL_WITH_LINE(err, "ERROR: negative CFL condition");
        return;
    }

    // Divergence
    dt->next = new Dt();
    dt = dt->next;
    hydro::kernel::getDtDiv(sizes.nel, hydro.div_sf, a1, a3, b1, b3, elvolume,
            cnu, cnv, scratch, dt->rdt, dt->idt, dt->sdt);
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
