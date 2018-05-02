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
#include "packages/hydro/driver/get.h"

#include <cassert>

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "packages/hydro/config.h"
#include "packages/hydro/kernel/get_artificial_viscosity.h"
#include "common/timer_control.h"
#include "common/data_control.h"
#include "common/sizes.h"
#include "utilities/data/global_configuration.h"

#include "utilities/comms/config.h"
#ifdef BOOKLEAF_MPI_SUPPORT
#include "utilities/comms/exchange.h"
#endif



namespace bookleaf {
namespace hydro {
namespace driver {

void
getArtificialViscosity(
        hydro::Config const &hydro,
        Sizes const &sizes,
        TimerControl &timers,
        DataControl &data)
{
    using constants::NCORN;
    using constants::NFACE;

    ScopedTimer st(timers, TimerID::GETVISCOSITY);

    int const nel = sizes.nel;

    auto cnx       = data[DataID::CNX].chost<double, VarDim, NCORN>();
    auto cny       = data[DataID::CNY].chost<double, VarDim, NCORN>();
    auto cnu       = data[DataID::TIME_CNU].chost<double, VarDim, NCORN>();
    auto cnv       = data[DataID::TIME_CNV].chost<double, VarDim, NCORN>();
    auto ndtype    = data[DataID::INDTYPE].chost<int, VarDim>();
    auto elel      = data[DataID::IELEL].chost<int, VarDim, NFACE>();
    auto elnd      = data[DataID::IELND].chost<int, VarDim, NCORN>();
    auto elfc      = data[DataID::IELFC].chost<int, VarDim, NFACE>();
    auto eldensity = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elcs2     = data[DataID::ELCS2].chost<double, VarDim>();
    auto elvisc    = data[DataID::ELVISC].host<double, VarDim>();
    auto dx        = data[DataID::TIME_DX].host<double, VarDim, NFACE>();
    auto dy        = data[DataID::TIME_DY].host<double, VarDim, NFACE>();
    auto du        = data[DataID::TIME_DU].host<double, VarDim, NFACE>();
    auto dv        = data[DataID::TIME_DV].host<double, VarDim, NFACE>();
    auto edviscx   = data[DataID::CNVISCX].host<double, VarDim, NFACE>();
    auto edviscy   = data[DataID::CNVISCY].host<double, VarDim, NFACE>();
    auto store     = data[DataID::TIME_STORE].host<double, VarDim, NCORN>();

    // XXX Missing code here that can't be merged

    // Initialisation
    kernel::initArtificialViscosity(cnx, cny, cnu, cnv, elvisc, dx, dy, du, dv,
            edviscx, edviscy, nel);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (hydro.comm->nproc > 1) {
        Error err;
        comms::exchange(*hydro.comm, 0, TimerID::COMMT, timers, err);
        if (err.failed()) {
            assert(false && "unhandled error");
        }
    }
#endif

    auto cdx = data[DataID::TIME_DX].chost<double, VarDim, NFACE>();
    auto cdy = data[DataID::TIME_DY].chost<double, VarDim, NFACE>();
    auto cdu = data[DataID::TIME_DU].chost<double, VarDim, NFACE>();
    auto cdv = data[DataID::TIME_DV].chost<double, VarDim, NFACE>();

    // Christensen's monotonic limit
    kernel::limitArtificialViscosity(sizes.nel, hydro.global->zerocut,
            hydro.cvisc1, hydro.cvisc2, ndtype, elel, elnd, elfc, eldensity,
            elcs2, cdu, cdv, cdx, cdy, store, edviscx, edviscy, elvisc);

    // Final Q calculation
    kernel::getArtificialViscosity(hydro.global->zerocut, cnx, cny, cnu, cnv,
            edviscx, edviscy, nel);
}

} // namespace driver
} // namespace hydro
} // namespace bookleaf
