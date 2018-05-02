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
#include "packages/check/driver/tests.h"

#ifdef BOOKLEAF_MPI_SUPPORT
#include <typhon.h>
#endif

#include "common/constants.h"
#include "common/config.h"
#include "common/runtime.h"
#include "common/timestep.h"
#include "common/sizes.h"
#include "common/data_control.h"

#include "utilities/data/gather.h"
#include "utilities/comms/config.h"

#include "packages/hydro/config.h"
#include "packages/check/kernel/tests.h"



namespace bookleaf {
namespace check {
namespace driver {

double
testSod(
#ifdef BOOKLEAF_MPI_SUPPORT
        Config const &config,
#else
        Config const &config __attribute__((unused)),
#endif
        Runtime const &runtime,
        DataControl &data)
{
    using constants::NCORN;

    // Gather co-ordinates
    utils::driver::cornerGather(*runtime.sizes, DataID::NDX, DataID::CNX, data);
    utils::driver::cornerGather(*runtime.sizes, DataID::NDY, DataID::CNY, data);

    double _basis;
    double _l1[2];
    View<double, 1> basis(&_basis);
    View<double, 2> l1(_l1);

    auto elvolume  = data[DataID::ELVOLUME].chost<double, VarDim>();
    auto eldensity = data[DataID::ELDENSITY].chost<double, VarDim>();
    auto elenergy  = data[DataID::ELENERGY].chost<double, VarDim>();
    auto cnx       = data[DataID::CNX].chost<double, VarDim, NCORN>();
    auto cny       = data[DataID::CNY].chost<double, VarDim, NCORN>();

    // Calculate L1 norm components for density and energy
    kernel::testSod(runtime.sizes->nel, runtime.timestep->time, elvolume,
            eldensity, elenergy, cnx, cny, basis, l1);

#ifdef BOOKLEAF_MPI_SUPPORT
    if (config.hydro->comm->nproc > 1) {
        double btotal;
        double ltotal[2];
        int const ldim = 2;

        int typh_err = TYPH_Reduce(basis.data(), nullptr, 0, &btotal, TYPH_OP_SUM);
        if (typh_err != TYPH_SUCCESS) {
            assert(false && "unhandled error");
        }

        typh_err = TYPH_Reduce(l1.data(), &ldim, 1, ltotal, TYPH_OP_SUM);
        if (typh_err != TYPH_SUCCESS) {
            assert(false && "unhandled error");
        }

        basis(0) = btotal;
        l1(0) = ltotal[0];
        l1(1) = ltotal[1];
    }
#endif

    // Use L1 norm of density (energy also calculated)
    return l1(0) / basis(0);
}

} // namespace driver
} // namespace check
} // namespace bookleaf
